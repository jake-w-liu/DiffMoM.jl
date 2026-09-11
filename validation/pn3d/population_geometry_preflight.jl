include("pilot.jl")
using CSV

function population_geometry_preflight(population_dir, output_path)
    manifest = JSON.parsefile(joinpath(population_dir, "population_manifest.json"))
    parameters_path = joinpath(population_dir, "parameters.csv")
    bytes2hex(sha256(read(parameters_path))) == manifest["artifact_sha256"]["parameters.csv"] ||
        error("population parameters changed")
    records = NamedTuple[]
    for row in CSV.File(parameters_path)
        mesh = make_bent_slotted_panel(; panel_length=row.main_length_m,
            panel_width=row.main_width_m, flange_width=row.flange_length_m,
            bend_angle=row.bend_angle_rad, slot_length=row.slot_length_m,
            slot_width=row.slot_width_m, max_edge=0.05)
        rwg = build_rwg(mesh)
        triangles = ntriangles(mesh)
        boundary_edges = 3triangles - 2rwg.nedges
        boundary_edges >= 0 || error("negative boundary-edge count")
        # Uniform midpoint refinement multiplies faces by four and subdivides
        # each boundary edge in two. Euler edge incidence then fixes RWG counts.
        counts = [(3triangles * 4^level - boundary_edges * 2^level) ÷ 2 for level in 0:3]
        counts[1] == rwg.nedges || error("coarse edge-incidence identity failed")
        m = counts[3] - counts[2]
        push!(records, (; case_id=String(row.case_id), split=String(row.split),
            coarse_unknowns=counts[1], level1_unknowns=counts[2],
            level2_unknowns=counts[3], level3_unknowns=counts[4],
            level1_exact_residual_terms=BigInt(counts[2]) * (counts[2]+1),
            level1_conditioning_work_bytes=DiffMoM._conditioning_work_bytes(m, 72)))
    end
    length(records) == manifest["training_cases"] + manifest["calibration_cases"] + manifest["test_cases"] ||
        error("population count differs from its manifest")
    mkpath(dirname(output_path))
    CSV.write(output_path, records)
    for key in (:coarse_unknowns, :level1_unknowns, :level2_unknowns, :level3_unknowns,
                :level1_exact_residual_terms, :level1_conditioning_work_bytes)
        values = getproperty.(records, key)
        println(key, ": ", minimum(values), " .. ", maximum(values))
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 2 || error("usage: population_geometry_preflight.jl POPULATION OUTPUT_CSV")
    population_geometry_preflight(abspath(ARGS[1]), abspath(ARGS[2]))
end
