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

println("DiffMoM experiment smoke passed: plate mesh, RWG basis, and patch mass are consistent.")
