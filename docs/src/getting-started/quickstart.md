# Quickstart

## Purpose

Run a minimal end-to-end workflow in a few minutes:
mesh → RWG basis → EFIE solve → field postprocessing → objective.

---

## Learning Goals

After this chapter, you should be able to:

1. Run a small PEC forward solve on a plate.
2. Compute scattered near-field and total electric field samples at off-surface points.
3. Compute far-field quantities and energy diagnostics.
4. Understand where to go next for optimization and validation.

---

## 1) Minimal Forward Solve (PEC Plate)

```julia
using DiffMoM

# Geometry and basis
mesh = make_rect_plate(0.1, 0.1, 6, 6)   # 1λ×1λ at 3 GHz (approx)
rwg  = build_rwg(mesh)

# Frequency
f  = 3e9
c0 = 299792458.0
λ0 = c0 / f
k  = 2π / λ0
η0 = 376.730313668

# EFIE and excitation
Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=η0)
k_vec = Vec3(0.0, 0.0, -k)
pol   = Vec3(1.0, 0.0, 0.0)
pw = make_plane_wave(k_vec, 1.0, pol)
v = assemble_excitation(mesh, rwg, pw; quad_order=3)

# Solve currents
I = solve_forward(Z, v)
```

---

## 2) Near Field and Total Field Samples

```julia
obs = [Vec3(0.0, 0.0, 0.15), Vec3(0.02, 0.00, 0.18)]
E_sca = compute_nearfield(mesh, rwg, I, obs, k; quad_order=3, eta0=η0)
E_tot = compute_total_field(mesh, rwg, I, pw, obs, k; quad_order=3, eta0=η0)

println("Scattered field at obs[1] = ", E_sca[:, 1])
println("Total field at obs[1]     = ", E_tot[:, 1])
```

`compute_nearfield` returns the scattered electric field only. `compute_total_field`
adds the pointwise incident field from the supplied excitation object, so it
requires a supported `AbstractExcitation` such as `PlaneWaveExcitation`.
For an analytical benchmark of this workflow, see the
[Near/Total-Field Rayleigh Sphere chapter](../validation/06-near-total-field-rayleigh-sphere.md).

---

## 3) Far Field and Energy Check

```julia
grid  = make_sph_grid(36, 72)
Gmat  = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=η0)
Eff   = compute_farfield(Gmat, I, length(grid.w))

Prad = radiated_power(Eff, grid; eta0=η0)
Pin  = input_power(I, v)
println("Energy ratio = ", Prad / Pin)
```

For a PEC scatterer, `Prad/Pin` should be close to 1 (up to discretization and
quadrature error).

---

## 4) Build a Directional Objective Quickly

```julia
pol_ff = pol_linear_x(grid)
mask = cap_mask(grid; theta_max=30 * π / 180)
Q = build_Q(Gmat, grid, pol_ff; mask=mask)
J = compute_objective(I, Q)   # J = real(I' * Q * I)
println("Directional objective = ", J)
```

This is the same `Q`-matrix objective form used by adjoint optimization.

---

## 5) Where to Go Next

- **Forward/mesh fundamentals**: Part II.
- **Adjoint + optimization**: Part III.
- **Validation and benchmarks**: Part V.
- **Runnable full workflows**: Tutorials section.

---

## 6) High-Value Example Commands

```bash
julia --project=. examples/03_beamsteering_physical_unitcell.jl
julia --project=. examples/04_pec_sphere_mie.jl
julia --project=. examples/21_near_total_field_rayleigh_sphere.jl
julia --project=. examples/06_aircraft_rcs.jl
```

---

## Code Mapping

- Forward assembly: `src/assembly/EFIE.jl`, `src/assembly/Excitation.jl`, `src/solver/Solve.jl`
- Field postprocessing and objectives: `src/postprocessing/NearField.jl`, `src/postprocessing/FarField.jl`, `src/optimization/QMatrix.jl`, `src/postprocessing/Diagnostics.jl`
- Example implementations: `examples/03_beamsteering_physical_unitcell.jl`, `examples/01_pec_plate_basics.jl`

---

## Exercises

- Basic: double `Ntheta`/`Nphi` and observe the energy-ratio change.
- Field check: verify numerically that `E_tot[:,1] - E_sca[:,1]` matches `plane_wave_field(obs[1], k_vec, 1.0, pol)`.
- Challenge: replace the PEC solve by an impedance-loaded solve using
  `assemble_full_Z` with a nonzero `theta` vector.
