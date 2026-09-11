include("pilot.jl")
using Profile

mesh = make_bent_slotted_panel(max_edge=EDGE)
source = source_wave()
k = 2pi * FREQUENCY / SPEED
prepared = solve_scattering(
    mesh, FREQUENCY, source; return_state=true, verbose=false,
    max_dense_matrix_bytes=WORK_BYTES)
pair = build_nested_rwg_pair(mesh; max_work_bytes=WORK_BYTES)
fine_matrix = assemble_Z_efie(pair.fine_mesh, pair.fine_rwg, k)
fine_rhs = assemble_excitation(pair.fine_mesh, pair.fine_rwg, source)
system = prepare_galerkin_error(
    prepared.state, pair, fine_matrix, fine_rhs; max_work_bytes=WORK_BYTES)
output_map = rcs_output_map(
    pair.fine_mesh, pair.fine_rwg, make_sph_grid(3, 4), k;
    max_work_bytes=WORK_BYTES, max_exact_work=RADIATION_EXACT_WORK)
model = condition_discretization_error(system, Int[]; tau=0.001, max_work_bytes=WORK_BYTES)
evaluate_error_outputs(model, output_map; max_work_bytes=WORK_BYTES)
Profile.clear()
elapsed = @elapsed Profile.@profile evaluate_error_outputs(
    model, output_map; max_work_bytes=WORK_BYTES)
println("PROFILED_OUTPUT_SECONDS=", elapsed)
Profile.print(format=:flat, sortedby=:count, mincount=100)

# Compare sparse-triangular matrix and column solves on the actual prior
# factor. Keep both answers for an independent numerical parity check.
lower = model.conditioning.lower
rhs = randn(MersenneTwister(1), ComplexF64, size(lower, 1), 24)
lower \ rhs
function column_solve(lower, rhs)
    result = similar(rhs)
    for column in axes(rhs, 2)
        result[:, column] .= lower \ Vector(rhs[:, column])
    end
    return result
end
column_solve(lower, rhs)
matrix_time = @elapsed matrix_solution = lower \ rhs
column_time = @elapsed column_solution = column_solve(lower, rhs)
println(JSON.json((
    triangular_matrix_s=matrix_time, triangular_columns_s=column_time,
    relative_difference=norm(matrix_solution - column_solution) / norm(matrix_solution))))
