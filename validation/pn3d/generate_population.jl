include("pilot.jl")
using CSV

function generate_panel_population(output)
    # The splits are independent whole cases; looks and mesh levels are not
    # additional samples. These counts define the numerical study population.
    splits = (("train", 60), ("calibration", 199), ("test", 300))
    # Fixed by the supplied pn_mom_3d/experiments/panel_study.yaml specification.
    seed = 20260907
    rng = MersenneTwister(seed)
    source = source_wave()
    direction = source.k_vec / norm(source.k_vec)
    grid = make_sph_grid(3, 4)
    look_count = size(grid.rhat, 2)
    minimum(norm(grid.rhat[:, i] - grid.rhat[:, j])
        for i in 1:look_count for j in i+1:look_count) > 1e-8 ||
        error("the population requires distinct look directions")
    parameters = NamedTuple[]
    for (split, count) in splits
        for index in 1:count
            # Independent uniform draws on the bounded geometric profile.
            beta = (60 + 60rand(rng)) * pi / 180
            slot_length = 0.08 + 0.08rand(rng)
            slot_width = 0.01 + 0.02rand(rng)
            push!(parameters, (; case_id=split * "-" * lpad(string(index), 4, '0'),
                split, index, bend_angle_rad=beta, slot_length_m=slot_length,
                slot_width_m=slot_width, main_length_m=0.40, main_width_m=0.30,
                flange_length_m=0.15, frequency_hz=FREQUENCY, c0_m_s=SPEED,
                incident_amplitude_v_m=source.E0,
                incident_dx=direction[1], incident_dy=direction[2], incident_dz=direction[3],
                polarization_x=source.pol[1], polarization_y=source.pol[2],
                polarization_z=source.pol[3]))
        end
    end
    length(unique(row.case_id for row in parameters)) == length(parameters) ||
        error("case identifiers are not unique")
    all(row -> pi/3 <= row.bend_angle_rad <= 2pi/3 &&
        0.08 <= row.slot_length_m <= 0.16 && 0.01 <= row.slot_width_m <= 0.03,
        parameters) || error("generated geometry is outside the declared population")
    looks = [(; look_id=j, theta_rad=grid.theta[j], phi_rad=grid.phi[j],
        direction_x=grid.rhat[1, j], direction_y=grid.rhat[2, j], direction_z=grid.rhat[3, j])
        for j in 1:look_count]
    # A broad logarithmic family of illustrative numerical masks is registered
    # before any held-out reference is evaluated, rather than selected by test outcomes.
    masks = [(; mask_id="mask-" * lpad(string(i), 2, '0'), look_id=j,
        threshold_m2=10.0^exponent) for (i, exponent) in enumerate(range(-3, 2; length=21))
        for j in 1:look_count]

    mkpath(dirname(output))
    mkdir(output)  # Refuse to replace an existing population, including a partial run.
    artifacts = ("parameters.csv" => parameters, "looks.csv" => looks, "masks.csv" => masks)
    hashes = Dict{String,String}()
    for (name, rows) in artifacts
        path = joinpath(output, name)
        CSV.write(path, rows)
        hashes[name] = bytes2hex(sha256(read(path)))
    end
    manifest = (; schema_version=1, seed, julia_version=string(VERSION),
        generator_sha256=bytes2hex(sha256(read(@__FILE__))),
        distribution_id="bent-slotted-panel-uniform-v1",
        distribution="independent uniform bend angle, slot length, and slot width",
        training_cases=60, calibration_cases=199, test_cases=300,
        look_count, mask_count=21, mask_interpretation="illustrative numerical requirements",
        bounds=(; bend_angle_degrees=[60, 120], slot_length_m=[0.08, 0.16],
            slot_width_m=[0.01, 0.03]), artifact_sha256=hashes)
    open(joinpath(output, "population_manifest.json"), "w") do io
        JSON.print(io, manifest, 2)
    end
    println("Generated ", length(parameters), " whole-case parameter sets, ",
        look_count, " looks, and 21 masks. No scattering references were evaluated.")
    return manifest
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 1 || error("usage: generate_population.jl NEW_OUTPUT_DIRECTORY")
    generate_panel_population(abspath(ARGS[1]))
end
