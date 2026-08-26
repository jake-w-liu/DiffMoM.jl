# Tutorial: PEC sphere RCS against Mie theory

A PEC sphere provides an analytical reference for the complete mesh, RWG,
EFIE, solve, far-field, and RCS path. The repository contains a compact example
and a detailed validation driver.

## Run the bounded example

From the project root:

```bash
julia --project=. examples/04_pec_sphere_mie.jl
```

The example generates an icosphere, solves it, samples a bounded spherical grid,
and compares one azimuth cut with `mie_bistatic_rcs_pec`. The `subdiv`, `a`,
`freq`, and `grid` assignments in `examples/04_pec_sphere_mie.jl` own the
executable setup. It writes no files.

The command exits nonzero unless its energy-ratio, cut-error, and backscatter
checks pass. The `MIE_EXAMPLE_MAX_*` constants in
`examples/04_pec_sphere_mie.jl` own the executable limits and appear in each
failure message.

These thresholds belong to this fixed mesh, frequency, and sampling setup. A
changed electrical size needs its own mesh and angular convergence evidence.

## Run the detailed validator

```bash
julia --project=. validation/mie/validate_mie_rcs.jl
```

This driver generates and re-imports the same sphere through STL, checks that it
is closed, solves the dense EFIE system, compares 360-point phi-zero and
phi-90-degree cuts, and evaluates power balance on a `60 x 36` full-sphere grid.

It adds solve-residual and two-cut comparison gates. Their executable values are
the `MIE_MAX_*` constants in `validation/mie/validate_mie_rcs.jl`; the
verification output prints those same values.

The detailed outputs are:

- `data/mie_rcs_phi0.csv`
- `data/mie_rcs_phi90.csv`
- `data/mie_rcs_summary.csv`
- `data/figs/mie_rcs_phi0.png`
- `data/figs/mie_rcs_phi90.png`
- `validation/mie/figs/mie_rcs_both_cuts.png`
- `validation/mie/figs/mie_rcs_error.png`

## Assemble the comparison manually

The following fragments show the same data flow when adapting the benchmark.

### 1. Load the tracked sphere

```julia
using DiffMoM
using LinearAlgebra
using Statistics

mesh = read_stl_mesh("validation/mie/sphere_ka2.1.stl")
assert_mesh_quality(mesh; allow_boundary=false, require_closed=true)
rwg = build_rwg(mesh; allow_boundary=false, require_closed=true)
```

The tracked STL has the benchmark geometry. The example instead creates the
icosphere locally, which avoids relying on a generated file.

### 2. Set the wave and solve

```julia
freq = 2.0e9
c0 = 299792458.0
k = 2pi * freq / c0
radius = 0.05

k_vec = Vec3(0.0, 0.0, -k)
khat_inc = k_vec / norm(k_vec)
pol = Vec3(1.0, 0.0, 0.0)

Z = assemble_Z_efie(mesh, rwg, k; quad_order=3)
v = assemble_excitation(
    mesh,
    rwg,
    make_plane_wave(k_vec, 1.0, pol);
    quad_order=3,
)
I = solve_forward(Z, v)

relative_residual = norm(Z * I - v) / norm(v)
isfinite(relative_residual) || error("non-finite solve residual")
```

The incidence direction is `-z`. Backscatter therefore points along `+z`, or
`-khat_inc`. Forward scatter points along `-z`, or `khat_inc`.

### 3. Compute MoM and Mie RCS on identical directions

```julia
grid = make_sph_grid(60, 36)
G = radiation_vectors(mesh, rwg, grid, k; quad_order=3)
E_ff = compute_farfield(G, I, length(grid.w))
sigma_mom = bistatic_rcs(E_ff; E0=1.0)

sigma_mie = [
    mie_bistatic_rcs_pec(
        k,
        radius,
        khat_inc,
        pol,
        Vec3(grid.rhat[:, q]),
    )
    for q in eachindex(grid.w)
]
```

`make_sph_grid` stores cell midpoints. A `60 x 36` grid therefore contains
theta centers at 1.5, 4.5, and so on, and phi centers at 5, 15, and so on. Do
not label its first azimuth cut as exactly zero degrees.

### 4. Extract a sampled cut

```julia
phi_target = minimum(grid.phi)
cut = findall(q -> abs(grid.phi[q] - phi_target) < 1e-12,
              eachindex(grid.phi))
sort!(cut; by=q -> grid.theta[q])

mom_db = 10 .* log10.(max.(sigma_mom[cut], 1e-30))
mie_db = 10 .* log10.(max.(sigma_mie[cut], 1e-30))
delta_db = mom_db .- mie_db

mae_db = mean(abs.(delta_db))
rmse_db = sqrt(mean(abs2, delta_db))
max_error_db = maximum(abs.(delta_db))
```

The `1e-30` floor only bounds the logarithm. Retain the linear values when
interpreting errors near a null.

### 5. Check backscatter and power

```julia
sigma_back_mom = backscatter_rcs(
    E_ff,
    grid,
    khat_inc;
    E0=1.0,
)
sigma_back_mie = mie_bistatic_rcs_pec(
    k,
    radius,
    khat_inc,
    pol,
    -khat_inc,
)

P_in = input_power(I, v)
P_rad = radiated_power(E_ff, grid)
energy_ratio = P_rad / P_in
```

`backscatter_rcs` accepts the incident propagation direction and selects the
stored observation direction nearest its negative. Inspect
`sigma_back_mom.angular_error_deg` when changing the angular grid.

## Interpreting mismatches

| Observed difference | Checks that distinguish possible causes |
|---|---|
| Nearly constant dB offset | Radius, field amplitude, units, and RCS normalization |
| Different angular shape | Exact observation directions, mesh refinement, and quadrature sensitivity |
| Large spikes near minima | Linear-scale error and the chosen logarithmic floor |
| Different phi cuts on a sphere | Mesh centering and winding, polarization, and matching azimuth samples |
| Backscatter differs but the cut agrees | Nearest-grid angular error and backscatter direction convention |
| Small residual but poor Mie agreement | Discretization, geometry, or postprocessing rather than the linear solve alone |

A low algebraic residual proves only that the discrete system was solved. The
Mie comparison tests whether the selected discretization and observable agree
with the analytical sphere problem.

## Convergence study

For a modified case, record the actual maximum edge length, RWG count,
quadrature order, angular grid, residual, linear error, dB error, and power
ratio. Refine one dimension at a time. Claim an asymptotic rate only after the
measured points show an interval that supports it.

The analytical order used by `mie_bistatic_rcs_pec` is selected automatically
from `ka`. The function also accepts an explicit `nmax` when a separate Mie
truncation study is required.

## Source map

| Task | Source |
|---|---|
| Compact benchmark | `examples/04_pec_sphere_mie.jl` |
| Detailed benchmark and artifacts | `validation/mie/validate_mie_rcs.jl` |
| PEC Mie series | `src/postprocessing/Mie.jl` |
| Far-field assembly | `src/postprocessing/FarField.jl` |
| RCS and power diagnostics | `src/postprocessing/Diagnostics.jl` |
| Near- and total-field sphere benchmark | `examples/21_near_total_field_rayleigh_sphere.jl` |
