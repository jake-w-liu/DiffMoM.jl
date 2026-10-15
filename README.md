# DiffMoM.jl

`DiffMoM.jl` is a Julia package for Method-of-Moments electromagnetic
simulation and differentiable inverse design.

DiffMoM includes surface EFIE MoM for PEC and impedance
surfaces, 2D volume integral equation tools, 3D DDA/VIE-style material
scattering, dielectric surface integral equations, physical-optics/PTD
approximations, periodic EFIE utilities, topology optimization helpers, and
analytical/external validation workflows.

It also ships a shielded multilayer planar-circuit solver aimed at
Sonnet/RFIC-style problems: native `.son` project ingestion, fabrication
artwork import (Gerber, ODB++, DXF, GDS), FFT-accelerated rectangular-grid
and conformal triangular-mesh solvers, port calibration/de-embedding, and
rational model fitting with SPICE/Spectre export.

## Installation

Use a Julia version allowed by the `julia` compatibility entry in
[`Project.toml`](Project.toml).

From a local checkout

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
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
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
xyz = 0.05 .* [1.0 -1.0 -1.0  1.0;
               1.0 -1.0  1.0 -1.0;
               1.0  1.0 -1.0 -1.0]
tri = [1 1 1 2;
       2 4 3 4;
       3 2 4 3]
mesh_closed = TriMesh(xyz, tri)
rwg_closed = build_rwg(mesh_closed; allow_boundary=false, require_closed=true)
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
res_sie = solve_dielectric_sie_3d(
    mesh_closed, rwg_closed, k, 2.5 + 0im, pw; solver=:direct)
```

Native Sonnet `.son` project solve (FFT-accelerated):

```julia
proj = read_sonnet_project("coupon.son")
prob = sonnet_planar_problem(proj; freq=5e9)
res  = solve_planar_ufft(prob, 5e9; precondition=true)
res.s   # S-parameters at the declared ports
```

The conformal triangular-mesh path handles polygons that do not align
with the rectangular cell grid:

```julia
res = solve_sonnet_conformal("coupon.son", 10e9;
                             edge_size=6e-5, interior_size=1.2e-4)
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
- Shielded multilayer planar circuits: spectral-domain rectangular-grid MoM with UFFT acceleration, conformal triangular-mesh solves for arbitrary polygons, vias/volumes/bricks, wall/CPW/floating ports, and calibration-standard de-embedding (TRL-style FIX/NONE/LINK).
- Native Sonnet `.son` project files, technology/STF materialization, scalar variable files, SPICE/Spectre netlists, and component/FLOAT networks.
- Fabrication artwork import: Gerber RS-274X, ODB++ (including vendor symbols, thermals, D-Packs, barcodes), DXF, and GDS maps — with certified region, sweep, and exact-arithmetic membership engines.
- Rational fitting, passivity certification, and equivalent-circuit export for S-parameter responses.
- Validation scripts for internal consistency, Mie sphere benchmarks, Bempp-cl comparisons, Meep periodic comparisons, Sonnet Lite 18.53 and Ansys HFSS 2025.1 oracles, physical-optics checks, cost scaling, and robustness studies.

## Validation

Focused regression suite (also run on every push and pull request):

```bash
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
```

This runs package-quality checks, small independent scientific/planar solver
comparisons, and targeted numerical, ownership, and allocation regressions.
Fixtures are shared within each case; large native corpora, mesh-convergence
studies, Monte Carlo checks, and subprocess RSS probes run separately:

```bash
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test(test_args=["full"])'
```

The same comprehensive suite is available through the manually triggered
**Full validation** GitHub Actions workflow for release checks, native-reference
updates, and convergence or resource investigations.
Full validation runs sequential groups in fresh Julia processes to release
compiler state between groups; any failed group fails the full run.

Analytical and internal checks:

```bash
julia --project=. validation/mie/validate_mie_rcs.jl
julia --project=. validation/mie/validate_dielectric_mie_dda.jl
julia --project=. validation/po/validate_po_vs_pofacets.jl
julia --project=. validation/scaling/run_cost_scaling.jl
julia --project=. validation/robustness/run_robustness_sweep.jl
```

`validation/po/validate_po_vs_pofacets.jl` requires a user-supplied
`examples/demo_aircraft.obj`; that geometry is not included in this repository.
Set `DMOM_AIRCRAFT_OBJ` to use another path.
The optional MATLAB aircraft comparisons also require `POFACETS_DIR`, or an
explicit `POFACETS_AIRCRAFT_MAT`, and fail unless their generated Julia inputs
are present.

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
- `test/` - focused default regressions and an explicit full scientific/native-reference validation suite.
- `examples/` - runnable examples from basic PEC plates through periodic topology optimization and PO/PTD comparisons.
- `docs/` - Documenter.jl manual and API reference, including the Sonnet
  parity ledger (`docs/src/advanced/sonnet-parity.md`) recording
  feature-by-feature verification against Sonnet Lite, HFSS, and Nexxim.
- `validation/` - analytical, external-solver, paper, scaling, robustness, and consistency workflows.
- `data/` and `figures/` - generated-output locations created by examples and
  validation scripts when needed.

## License

This project is released under the MIT License. See `LICENSE`.
