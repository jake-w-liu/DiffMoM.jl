using DiffMoM
using LinearAlgebra

mesh = DiffMoM.make_rect_plate(0.1, 0.1, 1, 1)
rwg = DiffMoM.build_rwg(mesh)
@assert rwg.nedges > 0

partition = DiffMoM.assign_patches_grid(mesh; nx=1, ny=1, nz=1)
Mp = DiffMoM.precompute_patch_mass(mesh, rwg, partition)
@assert length(Mp) == 1
@assert size(Mp[1], 1) == rwg.nedges
@assert norm(Matrix(Mp[1])) > 0

frequency = 3e9
speed = 299792458.0
k = 2pi * frequency / speed
source = PlaneWaveExcitation(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
_, coarse = solve_scattering(mesh, frequency, source;
    method=:dense_direct, return_state=true, check_resolution=false, verbose=false)
pair = build_nested_rwg_pair(mesh)
fine_result, fine = solve_scattering(pair.fine_mesh, frequency, source;
    method=:dense_direct, return_state=true, check_resolution=false, verbose=false)
system = prepare_galerkin_error(coarse, pair, fine.operator, fine.rhs)
grid = make_sph_grid(2, 3)
G = rcs_output_map(pair.fine_mesh, pair.fine_rwg, grid, k)
prepared = prepare_error_outputs(system, G)
unresolved = size(pair.Q, 2)
model = condition_discretization_error(system, collect(1:unresolved); tau=1.0)
outputs = evaluate_error_outputs(model, prepared)
reference = G * fine_result.I_coeffs

# Observing the entire nonsingular Schur system determines the fine-space
# correction. The direct fine solve is independent of this block elimination.
@assert model.conditioning.rank == unresolved
@assert all(isfinite, reference)
@assert norm(outputs.mean - reference) <= 1e-9 * norm(reference)
@assert all(iszero, outputs.square_root)

println("DiffMoM experiments passed: mesh, patch mass, and full-information field recovery.")
