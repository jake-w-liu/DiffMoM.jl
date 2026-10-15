module ValidationConsistencyGateTests
using Test, CSV, DataFrames
include(joinpath(@__DIR__, "..", "validation", "consistency", "generate_consistency_report.jl"))

@testset "Consistency report requires conservative energy and actual beam cases" begin
    mktempdir() do directory
        save(name, table) = CSV.write(joinpath(directory, name), table)
        energy(ratios) = save("convergence_study.csv",
            DataFrame(energy_ratio=ratios, max_grad_err=zeros(length(ratios))))
        energy([0.99, 1.01])
        save("gradient_verification.csv", DataFrame(rel_error=[0.]))
        save("robustness_sweep.csv", DataFrame(case=["f_nom", "f_+2pct"],
            J_opt_pct=[20., 19.], J_pec_pct=[10., 10.]))
        save("cost_scaling.csv", DataFrame(assembly_s=[1., 2.],
            solve_s=[1., 2.], opt_iter_s=[1., 2.]))
        save("beam_steer_trace.csv", DataFrame(iter=[1]))
        save("beam_steer_impedance.csv", DataFrame(patch=[1]))
        save("beam_steer_cut_phi0.csv", DataFrame(theta_deg=[0., 30.],
            dir_opt_dBi=[2., 3.], dir_pec_dBi=[1., 2.]))
        for (name, column) in (("beam_steer_farfield.csv", :dir_pec_dBi),
                ("bempp_pec_farfield.csv", :dir_bempp_dBi),
                ("julia_impedance_farfield.csv", :dir_julia_imp_dBi),
                ("bempp_impedance_farfield.csv", :dir_bempp_imp_dBi))
            table = DataFrame(theta_deg=[0., 30.], phi_deg=zeros(2))
            table[!, column] = [1., 2.]
            save(name, table)
        end
        beam = DataFrame(pass_main_theta_le_3deg=[true],
            pass_main_level_le_1p5db=[true], pass_sll_le_3db=[true])
        save("impedance_validation_matrix_summary.csv", beam)
        require_inputs(directory)
        @test all(last, consistency_checks(compute_paper_metrics(directory)))

        # A conservative minimum must not hide excess power at another mesh.
        energy([0.99, 2.])
        metrics = compute_paper_metrics(directory)
        @test metrics.min_energy_ratio == 0.99
        @test !all(last, consistency_checks(metrics))

        energy([0.99, 1.01])
        save("impedance_validation_matrix_summary.csv", beam[1:0, :])
        metrics = compute_paper_metrics(directory)
        checks = consistency_checks(metrics)
        @test !all(last, checks)
        report = joinpath(directory, "report.md")
        write_report_markdown(metrics, checks, report; commit="fixture")
        @test occursin("Beam-centric matrix status: FAIL", read(report, String))
    end
end
end
