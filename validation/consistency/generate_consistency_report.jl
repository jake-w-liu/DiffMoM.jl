#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using Printf
using Statistics

const REQUIRED_INPUTS = [
    ("convergence_study.csv", "julia --project=. validation/paper/run_convergence_study.jl"),
    ("gradient_verification.csv", "julia --project=. validation/paper/run_convergence_study.jl"),
    ("robustness_sweep.csv", "julia --project=. validation/robustness/run_robustness_sweep.jl"),
    ("cost_scaling.csv", "julia --project=. validation/scaling/run_cost_scaling.jl"),
    ("beam_steer_trace.csv", "julia --project=. validation/paper/run_beam_steering_case.jl"),
    ("beam_steer_cut_phi0.csv", "julia --project=. validation/paper/run_beam_steering_case.jl"),
    ("beam_steer_farfield.csv", "julia --project=. validation/paper/run_beam_steering_case.jl"),
    ("beam_steer_impedance.csv", "julia --project=. validation/paper/run_beam_steering_case.jl"),
    ("bempp_pec_farfield.csv", "python validation/bempp/run_pec_cross_validation.py"),
    ("bempp_impedance_farfield.csv", "python validation/bempp/run_impedance_validation_matrix.py"),
    ("julia_impedance_farfield.csv", "python validation/bempp/run_impedance_validation_matrix.py"),
]

function require_inputs(datadir::AbstractString)
    missing = String[]
    for (filename, command) in REQUIRED_INPUTS
        path = joinpath(datadir, filename)
        isfile(path) || push!(missing, "  - $(filename)  <- run `$(command)`")
    end

    imp_matrix_csv = joinpath(datadir, "impedance_validation_matrix_summary.csv")
    imp_matrix_csv_alt = joinpath(datadir, "impedance_validation_matrix_summary_paper_default.csv")
    if !(isfile(imp_matrix_csv) || isfile(imp_matrix_csv_alt))
        push!(missing, "  - impedance_validation_matrix_summary.csv  <- run `python validation/bempp/run_impedance_validation_matrix.py`")
    end

    isempty(missing) || error(
        "Missing required validation artifacts:\n" * join(missing, "\n")
    )
end

function nearest_theta_stats(theta_deg::AbstractVector{<:Real}, values::AbstractVector{<:Real}, target_deg::Float64)
    theta_unique = unique(theta_deg)
    nearest = theta_unique[argmin(abs.(theta_unique .- target_deg))]
    idx = findall(t -> abs(t - nearest) < 1e-12, theta_deg)
    return (
        target_theta_deg = target_deg,
        nearest_theta_deg = nearest,
        mean_value = mean(values[idx]),
        max_value = maximum(values[idx]),
    )
end

function crossval_metrics(
    df_ref::DataFrame,
    ref_col::Symbol,
    df_cmp::DataFrame,
    cmp_col::Symbol;
    target_theta_deg::Float64 = 30.0,
)
    left = select(df_ref, :theta_deg, :phi_deg, ref_col)
    right = select(df_cmp, :theta_deg, :phi_deg, cmp_col)
    merged = innerjoin(left, right, on=[:theta_deg, :phi_deg])

    Δ = merged[!, cmp_col] .- merged[!, ref_col]
    absΔ = abs.(Δ)

    phi_dist = min.(merged.phi_deg, 360 .- merged.phi_deg)
    idx_phi0 = findall(phi_dist .<= 2.5 + 1e-10)
    Δ_phi0 = Δ[idx_phi0]
    absΔ_phi0 = absΔ[idx_phi0]

    broadside = nearest_theta_stats(merged.theta_deg, absΔ, 0.0)
    target = nearest_theta_stats(merged.theta_deg, absΔ, target_theta_deg)

    return (
        num_common_points = nrow(merged),
        global_mean_diff_db = mean(Δ),
        global_mean_abs_diff_db = mean(absΔ),
        global_rmse_db = sqrt(mean(abs2, Δ)),
        global_max_abs_diff_db = maximum(absΔ),
        phi0_num_points = length(idx_phi0),
        phi0_mean_diff_db = mean(Δ_phi0),
        phi0_mean_abs_diff_db = mean(absΔ_phi0),
        phi0_rmse_db = sqrt(mean(abs2, Δ_phi0)),
        phi0_max_abs_diff_db = maximum(absΔ_phi0),
        broadside_nearest_theta_deg = broadside.nearest_theta_deg,
        broadside_mean_abs_diff_db = broadside.mean_value,
        broadside_max_abs_diff_db = broadside.max_value,
        target_nearest_theta_deg = target.nearest_theta_deg,
        target_mean_abs_diff_db = target.mean_value,
        target_max_abs_diff_db = target.max_value,
    )
end

function compute_paper_metrics(datadir::AbstractString)
    convergence = CSV.read(joinpath(datadir, "convergence_study.csv"), DataFrame)
    gradient = CSV.read(joinpath(datadir, "gradient_verification.csv"), DataFrame)
    robustness = CSV.read(joinpath(datadir, "robustness_sweep.csv"), DataFrame)
    cost = CSV.read(joinpath(datadir, "cost_scaling.csv"), DataFrame)
    trace = CSV.read(joinpath(datadir, "beam_steer_trace.csv"), DataFrame)
    cut_phi0 = CSV.read(joinpath(datadir, "beam_steer_cut_phi0.csv"), DataFrame)

    pec_bempp = CSV.read(joinpath(datadir, "bempp_pec_farfield.csv"), DataFrame)
    pec_julia = CSV.read(joinpath(datadir, "beam_steer_farfield.csv"), DataFrame)
    imp_bempp = CSV.read(joinpath(datadir, "bempp_impedance_farfield.csv"), DataFrame)
    imp_julia = CSV.read(joinpath(datadir, "julia_impedance_farfield.csv"), DataFrame)
    imp_matrix_path = joinpath(datadir, "impedance_validation_matrix_summary.csv")
    if !isfile(imp_matrix_path)
        imp_matrix_path = joinpath(datadir, "impedance_validation_matrix_summary_paper_default.csv")
    end
    imp_matrix = CSV.read(imp_matrix_path, DataFrame)

    pec_cv = crossval_metrics(pec_julia, :dir_pec_dBi, pec_bempp, :dir_bempp_dBi)
    imp_cv = crossval_metrics(imp_julia, :dir_julia_imp_dBi, imp_bempp, :dir_bempp_imp_dBi)

    n_cases = nrow(imp_matrix)
    n_pass_main_theta = count(imp_matrix.pass_main_theta_le_3deg)
    n_pass_main_level = count(imp_matrix.pass_main_level_le_1p5db)
    n_pass_sll = count(imp_matrix.pass_sll_le_3db)

    nominal = robustness[findfirst(robustness.case .== "f_nom"), :]
    fplus2 = robustness[findfirst(robustness.case .== "f_+2pct"), :]

    nearest_target = nearest_theta_stats(cut_phi0.theta_deg, cut_phi0.dir_opt_dBi .- cut_phi0.dir_pec_dBi, 30.0)

    monotonic_assembly = all(diff(cost.assembly_s) .> 0)
    monotonic_solve = all(diff(cost.solve_s) .> 0)
    monotonic_iter = all(diff(cost.opt_iter_s) .> 0)

    return (
        max_grad_rel_err_reference = maximum(gradient.rel_error),
        max_grad_rel_err_mesh_sweep = maximum(convergence.max_grad_err),
        min_energy_ratio = minimum(convergence.energy_ratio),
        nominal_J_opt_pct = nominal.J_opt_pct,
        nominal_J_pec_pct = nominal.J_pec_pct,
        nominal_improvement_factor = nominal.J_opt_pct / nominal.J_pec_pct,
        gain_target_db = nearest_target.mean_value,
        gain_target_nearest_theta_deg = nearest_target.nearest_theta_deg,
        fplus2_J_opt_pct = fplus2.J_opt_pct,
        pec_crossval = pec_cv,
        imp_crossval = imp_cv,
        beam_matrix_num_cases = n_cases,
        beam_matrix_pass_main_theta = n_pass_main_theta,
        beam_matrix_pass_main_level = n_pass_main_level,
        beam_matrix_pass_sll = n_pass_sll,
        monotonic_assembly = monotonic_assembly,
        monotonic_solve = monotonic_solve,
        monotonic_iter = monotonic_iter,
    )
end

function consistency_checks(metrics)
    return [
        ("Mesh-sweep max gradient error <= 3e-6", metrics.max_grad_rel_err_mesh_sweep <= 3e-6),
        ("Minimum energy ratio >= 0.98", metrics.min_energy_ratio >= 0.98),
        ("Nominal J_opt > nominal J_PEC", metrics.nominal_J_opt_pct > metrics.nominal_J_pec_pct),
        ("PEC near-target mean |ΔD| <= 0.5 dB", metrics.pec_crossval.target_mean_abs_diff_db <= 0.5),
        ("Beam-matrix |Δθ_main| <= 3° for all cases", metrics.beam_matrix_pass_main_theta == metrics.beam_matrix_num_cases),
        ("Beam-matrix |ΔD_main| <= 1.5 dB for all cases", metrics.beam_matrix_pass_main_level == metrics.beam_matrix_num_cases),
        ("Beam-matrix |ΔSLL| <= 3 dB for all cases", metrics.beam_matrix_pass_sll == metrics.beam_matrix_num_cases),
        ("Cost assembly increases with N", metrics.monotonic_assembly),
        ("Cost solve increases with N", metrics.monotonic_solve),
        ("Cost per-iteration increases with N", metrics.monotonic_iter),
    ]
end

function git_short_commit(repo_root::AbstractString)
    try
        return readchomp(`git -C $repo_root rev-parse --short HEAD`)
    catch
        return "unknown"
    end
end

function write_snapshot_csv(metrics, out_csv::AbstractString)
    rows = DataFrame(
        metric = String[],
        value = Float64[],
        unit = String[],
    )

    push!(rows, ("max_grad_rel_err_reference", metrics.max_grad_rel_err_reference, ""))
    push!(rows, ("max_grad_rel_err_mesh_sweep", metrics.max_grad_rel_err_mesh_sweep, ""))
    push!(rows, ("min_energy_ratio", metrics.min_energy_ratio, ""))
    push!(rows, ("nominal_J_opt_pct", metrics.nominal_J_opt_pct, "%"))
    push!(rows, ("nominal_J_pec_pct", metrics.nominal_J_pec_pct, "%"))
    push!(rows, ("nominal_improvement_factor", metrics.nominal_improvement_factor, "x"))
    push!(rows, ("gain_target_db", metrics.gain_target_db, "dB"))
    push!(rows, ("fplus2_J_opt_pct", metrics.fplus2_J_opt_pct, "%"))
    push!(rows, ("pec_global_mean_abs_diff_db", metrics.pec_crossval.global_mean_abs_diff_db, "dB"))
    push!(rows, ("pec_target_mean_abs_diff_db", metrics.pec_crossval.target_mean_abs_diff_db, "dB"))
    push!(rows, ("imp_global_mean_abs_diff_db", metrics.imp_crossval.global_mean_abs_diff_db, "dB"))
    push!(rows, ("imp_target_mean_abs_diff_db", metrics.imp_crossval.target_mean_abs_diff_db, "dB"))
    push!(rows, ("beam_matrix_num_cases", metrics.beam_matrix_num_cases, "count"))
    push!(rows, ("beam_matrix_pass_main_theta", metrics.beam_matrix_pass_main_theta, "count"))
    push!(rows, ("beam_matrix_pass_main_level", metrics.beam_matrix_pass_main_level, "count"))
    push!(rows, ("beam_matrix_pass_sll", metrics.beam_matrix_pass_sll, "count"))

    CSV.write(out_csv, rows)
end

function write_report_markdown(metrics, checks, out_md::AbstractString; commit::AbstractString)
    lines = String[]
    push!(lines, "# Paper Consistency Report")
    push!(lines, "")
    push!(lines, "- Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "- Repository commit: `$commit`")
    push!(lines, "")
    push!(lines, "## Check Summary")
    for (label, pass) in checks
        mark = pass ? "[PASS]" : "[FAIL]"
        push!(lines, "- $mark $label")
    end
    push!(lines, "")
    push!(lines, "## Key Metrics")
    push!(lines, "| Metric | Value |")
    push!(lines, "|---|---:|")
    push!(lines, @sprintf("| Max gradient rel. error (reference) | %.3e |", metrics.max_grad_rel_err_reference))
    push!(lines, @sprintf("| Max gradient rel. error (mesh sweep) | %.3e |", metrics.max_grad_rel_err_mesh_sweep))
    push!(lines, @sprintf("| Minimum energy ratio | %.3f |", metrics.min_energy_ratio))
    push!(lines, @sprintf("| Nominal J_opt | %.3f %% |", metrics.nominal_J_opt_pct))
    push!(lines, @sprintf("| Nominal J_PEC | %.3f %% |", metrics.nominal_J_pec_pct))
    push!(lines, @sprintf("| Nominal improvement factor | %.2f x |", metrics.nominal_improvement_factor))
    push!(lines, @sprintf("| Gain at target angle (nearest %.1f deg) | %.3f dB |", metrics.gain_target_nearest_theta_deg, metrics.gain_target_db))
    push!(lines, @sprintf("| J_opt at +2%% frequency | %.3f %% |", metrics.fplus2_J_opt_pct))
    push!(lines, "")
    push!(lines, "## External Cross-Validation Metrics")
    push!(lines, "| Case | Mean |ΔD| (global) | Mean |ΔD| near target |")
    push!(lines, "|---|---:|---:|")
    push!(lines, @sprintf("| PEC (Bempp vs Julia) | %.3f dB | %.3f dB |", metrics.pec_crossval.global_mean_abs_diff_db, metrics.pec_crossval.target_mean_abs_diff_db))
    push!(lines, @sprintf("| Impedance (Bempp vs Julia) | %.3f dB | %.3f dB |", metrics.imp_crossval.global_mean_abs_diff_db, metrics.imp_crossval.target_mean_abs_diff_db))
    push!(lines, "")
    push!(lines, "## Beam-Centric Matrix Gates")
    push!(lines, "- Cases with |Δθ_main| <= 3°: $(metrics.beam_matrix_pass_main_theta)/$(metrics.beam_matrix_num_cases)")
    push!(lines, "- Cases with |ΔD_main| <= 1.5 dB: $(metrics.beam_matrix_pass_main_level)/$(metrics.beam_matrix_num_cases)")
    push!(lines, "- Cases with |ΔSLL| <= 3 dB: $(metrics.beam_matrix_pass_sll)/$(metrics.beam_matrix_num_cases)")
    push!(lines, "- Beam-centric matrix status: $((metrics.beam_matrix_pass_main_theta == metrics.beam_matrix_num_cases && metrics.beam_matrix_pass_main_level == metrics.beam_matrix_num_cases && metrics.beam_matrix_pass_sll == metrics.beam_matrix_num_cases) ? "PASS" : "FAIL")")

    open(out_md, "w") do io
        for line in lines
            println(io, line)
        end
    end
end

function main()
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    datadir = joinpath(repo_root, "data")
    out_md = joinpath(datadir, "paper_consistency_report.md")
    out_csv = joinpath(datadir, "paper_metrics_snapshot.csv")

    require_inputs(datadir)
    metrics = compute_paper_metrics(datadir)
    checks = consistency_checks(metrics)
    commit = git_short_commit(repo_root)

    write_snapshot_csv(metrics, out_csv)
    write_report_markdown(metrics, checks, out_md; commit=commit)

    println("Saved $(relpath(out_csv, repo_root))")
    println("Saved $(relpath(out_md, repo_root))")
    for (label, pass) in checks
        mark = pass ? "PASS" : "FAIL"
        println("[$mark] $label")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
