# DiffMoM.jl

`DiffMoM.jl` is a Julia package for Method-of-Moments electromagnetic
simulation and differentiable inverse design. The package name is `DiffMoM`,
the module name is `DiffMoM`, and the current version is `0.1.0`.

The current implementation includes surface EFIE MoM for PEC and impedance
surfaces, 2D volume integral equation tools, 3D DDA/VIE-style material
scattering, dielectric surface integral equations, physical-optics/PTD
approximations, periodic EFIE utilities, topology optimization helpers, and
analytical/external validation workflows.

## Installation

From a local checkout:

```julia
import Pkg
Pkg.activate("path/to/DiffMoM.jl")
Pkg.instantiate()
using DiffMoM
```

From GitHub:

```julia
import Pkg
Pkg.add(url="https://github.com/jake-w-liu/DiffMoM.jl")
using DiffMoM
```

For the documentation environment:

```bash
julia --project=docs docs/make.jl
```

## Quick Start

Run from the package root:

```bash
julia --project=. test/runtests.jl
julia --project=. examples/01_pec_plate_basics.jl
julia --project=. examples/08_solve_scattering_workflow.jl
```

Minimal dense EFIE assembly:

```julia
using DiffMoM

mesh = make_rect_plate(0.1, 0.1, 6, 6)
rwg = build_rwg(mesh)
k = 2π / 0.1
Z = assemble_Z_efie(mesh, rwg, k; quad_order=3)
```

High-level scattering workflow:

```julia
using DiffMoM

freq = 3e9
c0 = 299792458.0
k = 2π * freq / c0
mesh = make_rect_plate(0.15, 0.15, 8, 8)
exc = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))

result = solve_scattering(mesh, freq, exc; method=:auto, verbose=true)
```

3D electric DDA/VIE-style material scattering:

```julia
using DiffMoM

k = 2π / 0.1
grid = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 7, 7, 7)
epsr = fill(2.5 + 0im, grid.nvoxels)
Einc = planewave_dda_3d(grid, Vec3(0.0, 0.0, k), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
res = solve_dda_3d(grid, k, epsr, Einc; solver=:gmres)
Ffar = farfield_dda_3d(res, Vec3(0.0, 1.0, 0.0))
```

Coupled electric-magnetic DDA:

```julia
Einc, Hinc = planewave_em_dda_3d(grid, Vec3(0.0, 0.0, k), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
res_em = solve_em_dda_3d(grid, k, 2.5 + 0im, 1.2 + 0im, Einc, Hinc; solver=:gmres)
```

Closed dielectric surface IE:

```julia
rwg_closed = build_rwg(mesh_closed; allow_boundary=false, require_closed=true)
A = assemble_pmchwt_3d(mesh_closed, rwg_closed, k, 2.5 + 0im)
res_sie = solve_dielectric_sie_3d(mesh_closed, rwg_closed, k, 2.5 + 0im, rhs;
                                  solver=:gmres)
```

## Implemented Capabilities

- Surface EFIE with RWG basis functions, triangle quadrature, singular/near-singular handling, impedance loading, and periodic EFIE corrections.
- Excitations for plane waves, dipoles, loops, monopoles, delta gaps, ports, pattern feeds, and multiple-source assembly.
- Dense direct solves, dense GMRES, matrix-free EFIE operators, ACA H-matrix operators, MLFMA operators, near-field sparse preconditioning, and adjoint GMRES.
- Mesh generation and I/O for rectangular/circular/parabolic surfaces plus OBJ, STL, and Gmsh MSH import/export, repair, coarsening, refinement, and resolution checks.
- Far-field, near-field, total-field, RCS, power/energy diagnostics, Mie-series references, physical optics, PTD edge diffraction, and visualization helpers.
- Differentiable design tools for impedance gradients, Q-matrix objectives, projected L-BFGS, directivity optimization, multi-angle RCS, density filtering/projection, and density adjoints.
- 2D TM VIE assembly/solves, 2D Green functions, line/plane-wave excitation, scattered-field Jacobians, and cylindrical Mie references.
- 3D DDA/VIE-style material solvers for isotropic, diagonal anisotropic, tensor anisotropic, magnetodielectric, explicit bianisotropic polarizability, and normalized bianisotropic constitutive models.
- Dense and matrix-free PMCHWT/Muller dielectric SIE assembly/solves for closed homogeneous isotropic surfaces.
- Validation scripts for internal consistency, Mie sphere benchmarks, Bempp-cl comparisons, Meep periodic comparisons, physical-optics checks, cost scaling, and robustness studies.

## Examples

All example commands are run from the package root.

```bash
julia --project=. examples/01_pec_plate_basics.jl
julia --project=. examples/02_impedance_optimization.jl
julia --project=. examples/03_beamsteering_physical_unitcell.jl
julia --project=. examples/04_pec_sphere_mie.jl
julia --project=. examples/05_solver_methods.jl
julia --project=. examples/05b_aca_scaling.jl
julia --project=. examples/07_pattern_feed.jl
julia --project=. examples/08_solve_scattering_workflow.jl
julia --project=. examples/09_mom_vs_po.jl
julia --project=. examples/12_plate_rcs_stl_roundtrip.jl
julia --project=. examples/13_sphere_rcs_optimization.jl
julia --project=. examples/14_periodic_to_validation.jl
julia --project=. examples/15_periodic_to_demo.jl
julia --project=. examples/16_periodic_to_mesh_convergence.jl
julia --project=. examples/17_periodic_to_beamsteer_demo.jl
julia --project=. examples/18_periodic_to_multistart_study.jl
julia --project=. examples/19_periodic_to_robustness_map.jl
julia --project=. examples/20_periodic_to_redistribution_demo.jl
julia --project=. examples/21_near_total_field_rayleigh_sphere.jl
julia --project=. examples/22_po_ptd_comparison.jl
julia --project=. examples/23_circular_plate_ptd.jl
```

Aircraft examples require `examples/demo_aircraft.obj`:

```bash
julia --project=. examples/06_aircraft_rcs.jl
julia -t 4 --project=. examples/09a_aircraft_po.jl
julia -t 4 --project=. examples/10_mlfma_scaling.jl
julia -t 4 --project=. examples/11_mlfma_finer.jl
```

## Validation

Regression suite:

```bash
julia --project=. test/runtests.jl
```

Analytical and internal checks:

```bash
julia --project=. validation/mie/validate_mie_rcs.jl
julia --project=. validation/mie/validate_dielectric_mie_dda.jl
julia --project=. validation/po/validate_po_vs_pofacets.jl
julia --project=. validation/scaling/run_cost_scaling.jl
julia --project=. validation/robustness/run_robustness_sweep.jl
```

Bempp-cl cross-validation requires Python, `bempp-cl`, and Gmsh:

```bash
julia --project=. validation/paper/run_beam_steering_case.jl
python validation/bempp/run_pec_cross_validation.py
python validation/bempp/compare_pec_to_julia.py
python validation/bempp/run_impedance_validation_matrix.py
python validation/bempp/sweep_impedance_conventions.py --run-julia
```

Meep periodic cross-validation requires a Python environment with the packages
listed under `validation/meep/requirements.txt`.

## Repository Layout

- `src/` - package source, module entrypoint, solvers, assembly, postprocessing, optimization, and material models.
- `test/` - sequential regression tests covering surface MoM, periodic topology, 2D/3D material solvers, FFT DDA, EM DDA, and dielectric SIE.
- `examples/` - runnable examples from basic PEC plates through periodic topology optimization and PO/PTD comparisons.
- `docs/` - Documenter.jl manual and API reference.
- `validation/` - analytical, external-solver, paper, scaling, robustness, and consistency workflows.
- `data/` and `figures/` - generated outputs from tests, examples, and validation scripts.

## Citation

If this package contributes to your work, please cite:

- J. W. Liu, *DiffMoM.jl: Open Differentiable Method-of-Moments Inverse-Design Pipeline*, GitHub repository, 2026.

## License

This project is released under the MIT License. See `LICENSE`.
