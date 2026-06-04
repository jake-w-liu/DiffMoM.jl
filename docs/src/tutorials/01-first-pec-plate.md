# Tutorial 1: First PEC Plate Scattering

## Purpose

Run your first physically meaningful Method‑of‑Moments (MoM) simulation: compute
the scattered field from a perfectly electrically conducting (PEC) plate
illuminated by a plane wave. This tutorial introduces the core workflow:

1. Geometry creation and RWG basis construction.
2. EFIE matrix assembly and excitation definition.
3. Direct solve and residual check.
4. Optional near-/total-field sampling plus far‑field computation and energy‑balance validation.

By the end, you will have verified that the basic MoM pipeline works correctly
on your system and produces results that pass fundamental sanity checks.

---

## Learning Goals

After this tutorial, you should be able to:

1. Create a simple plate mesh using `make_rect_plate` and build RWG basis functions.
2. Assemble the EFIE impedance matrix `Z` and plane‑wave excitation object / vector.
3. Solve the linear system `Z I = v` and compute the relative residual.
4. Sample the scattered near field and total electric field at off-surface points.
5. Compute the far‑field pattern and verify power conservation via `energy_ratio`.
6. Interpret the output numbers to identify potential issues (mesh defects, unit
   mistakes, insufficient quadrature).
7. Modify the script to change frequency, plate size, or discretization.

---

## 1) Quick Start: Run the Provided Example

The fastest way to see a working simulation is to execute the packaged convergence
study. From the repository root:

```bash
julia --project=. validation/paper/run_convergence_study.jl
```

This script performs a mesh‑refinement sweep on a PEC plate, solving at each
level and recording the residual, energy ratio, and condition number. Output is
written to `data/convergence_study.csv`. The script also prints a summary table
to the console.

**Expected console output** (abbreviated):

```
Convergence study for PEC plate
────────────────────────────────
N    residual       P_rad/P_in   cond_est
────────────────────────────────
64   3.2e-15        0.99998      2.1e+04
256  5.7e-15        0.99997      3.8e+05
...
```

If you see similar numbers (residual ∼1e‑14, energy ratio ∼1), your installation
is working correctly.

**What the script does**:

- Creates a square plate of side 0.1 m (10 cm) at 3 GHz.
- Varies the mesh density from coarse to fine.
- For each mesh, assembles the EFIE matrix, solves, computes far field, and
  checks energy balance.
- Writes the results to a CSV file for later analysis.

After verifying the script runs, proceed to the manual workflow to understand
each step.

---

## 2) Step‑by‑Step Manual Workflow

Open a Julia REPL in the project directory (or create a new script) and follow
these steps.

### Step 2.1: Load the Package

```julia
using DiffMoM
```

If you encounter an error, ensure you have activated the project with
`] activate .` and instantiated the environment with `] instantiate`.

### Step 2.2: Define Physical Parameters

```julia
f = 3e9                     # frequency (Hz)
c0 = 299792458.0            # speed of light (m/s)
λ = c0 / f                  # wavelength (m)
k = 2π / λ                  # wave number (rad/m)
η0 = 376.730313668          # free‑space impedance (Ω)
```

All lengths are in **meters**, frequencies in **hertz**. This is a common source
of mistakes—double‑check.

### Step 2.3: Create a Plate Mesh

```julia
mesh = make_rect_plate(0.1, 0.1, 4, 4)
```

This creates a rectangular plate in the xy‑plane, centered at the origin, with
side lengths `Lx = Ly = 0.1 m`. The plate is discretized with `4×4` cells in the
x‑ and y‑directions, resulting in `2×4×4 = 32` triangles and approximately
`(4+1)×(4+1) = 25` vertices.

### Step 2.4: Build RWG Basis Functions

```julia
rwg = build_rwg(mesh)
```

The Rao–Wilton–Glisson (RWG) basis functions are defined on each interior edge
of the mesh. The number of unknowns `N = rwg.nedges` is printed; for this mesh
you should see `N = 64`.

### Step 2.5: Assemble EFIE Matrix and Excitation Vector

```julia
Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=η0)
```

This forms the dense `N×N` impedance matrix using Gaussian quadrature of order
3 (default). Assembly time scales as `O(N²)`.

```julia
pw = make_plane_wave(Vec3(0,0,-k), 1.0, Vec3(1,0,0))
v = assemble_excitation(mesh, rwg, pw; quad_order=3)
```

The excitation object `pw` represents a plane wave propagating in the `-z`
direction with amplitude 1 V/m and x‑polarization. Keeping the excitation
object is useful later if you want total-field samples via `compute_total_field`.

### Step 2.6: Solve the Linear System

```julia
I = solve_forward(Z, v)
```

The package uses a dense LU factorization (via `LinearAlgebra.lu!`) to solve
`Z I = v`. The solution `I` is a complex vector of RWG coefficients.

### Step 2.7: Sample Near and Total Fields (Optional)

```julia
obs = [Vec3(0.0, 0.0, 0.15), Vec3(0.02, 0.0, 0.18)]
E_sca = compute_nearfield(mesh, rwg, I, obs, k; quad_order=3, eta0=η0)
E_tot = compute_total_field(mesh, rwg, I, pw, obs, k; quad_order=3, eta0=η0)

println("Scattered field at obs[1] = ", E_sca[:, 1])
println("Total field at obs[1]     = ", E_tot[:, 1])
```

This step is optional for a first run, but it is the recommended way to inspect
local field values. Observation points must stay off the surface.

For an analytical benchmark of these local-field routines, see the
[Near/Total-Field Rayleigh Sphere chapter](../validation/06-near-total-field-rayleigh-sphere.md)
and `examples/21_near_total_field_rayleigh_sphere.jl`.

### Step 2.8: Compute Far‑Field Pattern

```julia
grid = make_sph_grid(36, 72)      # 36 polar × 72 azimuth samples
G = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=η0)
E = compute_farfield(G, I, length(grid.w))
```

`grid` defines the angular directions where the far field is evaluated.
`radiation_vectors` computes the mapping from surface currents to far‑field
electric field. `compute_farfield` applies that mapping to the solution `I`.

### Step 2.9: Validate Results

```julia
residual = norm(Z * I - v) / norm(v)
println("Relative residual = ", residual)

ρ = energy_ratio(I, v, E, grid; eta0=η0)
println("Energy ratio P_rad / P_in = ", ρ)
```

For a lossless PEC scatterer, the energy ratio should be very close to 1 (within
a few percent). The residual should be near machine precision (∼1e‑14 for double
precision).

### Complete Script

Combine all steps into a single script (also available as
`examples/01_pec_plate_basics.jl`):

```julia
using DiffMoM

f = 3e9
c0 = 299792458.0
λ = c0 / f
k = 2π / λ
η0 = 376.730313668

mesh = make_rect_plate(0.1, 0.1, 4, 4)
rwg  = build_rwg(mesh)

Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=η0)
pw = make_plane_wave(Vec3(0,0,-k), 1.0, Vec3(1,0,0))
v = assemble_excitation(mesh, rwg, pw; quad_order=3)
I = solve_forward(Z, v)

obs = [Vec3(0.0, 0.0, 0.15), Vec3(0.02, 0.0, 0.18)]
E_sca = compute_nearfield(mesh, rwg, I, obs, k; quad_order=3, eta0=η0)
E_tot = compute_total_field(mesh, rwg, I, pw, obs, k; quad_order=3, eta0=η0)

grid = make_sph_grid(36, 72)
G = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=η0)
E = compute_farfield(G, I, length(grid.w))

println("Relative residual = ", norm(Z*I - v) / norm(v))
println("Energy ratio      = ", energy_ratio(I, v, E, grid; eta0=η0))
println("E_tot - E_sca     = ", E_tot[:, 1] - E_sca[:, 1])
```

---

## 3) Interpreting the Results

### Residual

The relative residual `‖Z I - v‖ / ‖v‖` measures how well the linear system is
solved. With a direct LU factorization and double‑precision arithmetic, you
should see:

- **Excellent**: `≤ 1e‑12`
- **Acceptable**: `≤ 1e‑8`
- **Suspicious**: `> 1e‑6`

If the residual is large, the matrix may be ill‑conditioned (check with
`condition_diagnostics(Z)`), or there may be a unit mistake (e.g., `k` wrong by
orders of magnitude).

### Energy Ratio

For a lossless PEC scatterer, the power radiated (or scattered) must equal the
power extracted from the incident field (power balance). The function
`energy_ratio` computes `P_rad / P_in`. Expect:

- **Ideal**: `0.99 ≤ ρ ≤ 1.01`
- **Tolerable**: `0.95 ≤ ρ ≤ 1.05`
- **Problematic**: `ρ < 0.9` or `ρ > 1.1`

If `ρ` is far from 1, possible causes are:

1. **Insufficient angular grid**: Increase `make_sph_grid(Nθ, Nφ)` resolution.
2. **Mesh defects**: Run `mesh_quality_report(mesh)`.
3. **Quadrature order too low**: Raise `quad_order` in `assemble_Z_efie` and
   `radiation_vectors` (default is 3).
4. **Electrical size too small** (`L/λ < 0.1`): The EFIE becomes ill‑conditioned
   at low frequencies; consider using a combined field integral equation (CFIE)
   if available.

### Condition Number

You can estimate the condition number with:

```julia
diag = condition_diagnostics(Z)
println("Condition estimate = ", diag.cond)
```

For a plate of moderate electrical size (`L/λ ≈ 1`), the condition number may be
`1e4–1e6`. If it exceeds `1e12`, the solve may become inaccurate.

### Near‑Field and Total‑Field Samples

For off-surface points, `compute_nearfield` gives the scattered field and
`compute_total_field` returns `E_inc + E_sca`. A quick sanity check is that
`E_tot - E_sca` should equal the incident plane wave at the same point. If it
does not, first verify that you passed the same excitation object used to build
the RHS.

### Far‑Field Pattern

Plot the far‑field pattern to visually verify it looks physically reasonable
(broadside scattering for a plate at normal incidence). Use your favorite
plotting package (e.g., `Plots.jl`) or examine the CSV output from the example
script.

---

## 4) Troubleshooting Common Issues

### “MethodError: no method matching assemble_Z_efie(…)”

- **Cause**: The function signature changed, or you are using an older version
  of the package.
- **Fix**: Check the current function signature with `?assemble_Z_efie` in the
  REPL. Ensure you have the latest commit.

### “OutOfMemoryError” during assembly

- **Cause**: The dense matrix size exceeds available RAM. For `N` unknowns, the
  matrix requires ≈ `16 N²` bytes.
- **Fix**: Reduce mesh size (`Nx`, `Ny`). Use `estimate_dense_matrix_gib(N)` to
  estimate memory before assembly. For complex geometries, coarsen the mesh with
  `coarsen_mesh_to_target_rwg`.

### Large residual (> 1e‑6)

- **Cause**: Ill‑conditioning, unit mistake, or corrupted assembly.
- **Diagnosis**:
  1. Compute `L/λ` (see Unit Debugging in the Debugging Playbook).
  2. Run `condition_diagnostics(Z)`.
  3. Enable GMRES solver: `I = solve_forward(Z, v; solver=:gmres)`.
- **Fix**: Correct unit mistakes, increase mesh coarseness, or enable
  preconditioning.

### Energy ratio far from 1

- **Cause**: Power imbalance due to numerical errors.
- **Diagnosis**:
  1. Increase angular grid density (e.g., `make_sph_grid(72, 144)`).
  2. Increase quadrature order (`quad_order=4`).
  3. Verify mesh quality (`mesh_quality_report`).
- **Fix**: Apply the above steps; if the problem persists, check for low‑frequency
  breakdown (if `L/λ < 0.1`).

### Far‑field pattern looks noisy or has spikes

- **Cause**: Insufficient quadrature for near‑singular integrals.
- **Fix**: Increase `quad_order` in `assemble_Z_efie` and `radiation_vectors`.

### “Non‑manifold edge” or “orientation conflict” errors

- **Cause**: Mesh defects, especially with imported OBJ files.
- **Fix**: Run `repair_mesh_for_simulation` before building RWG.

### Slow assembly time

- **Cause**: Assembly scales as `O(N²)`; high quadrature order increases constant factor.
- **Fix**: Use `quad_order=3` (default) unless high accuracy is required. For
  large `N`, consider coarsening.

### Still stuck?

Refer to the **Debugging Playbook** (Chapter 4 of Advanced Workflows) for a
systematic diagnostic order.

---

## 5) Code Mapping

| Task | Source File | Key Functions |
|------|-------------|---------------|
| Mesh creation | `src/geometry/Mesh.jl` | `make_rect_plate` |
| RWG basis | `src/basis/RWG.jl` | `build_rwg` |
| EFIE assembly | `src/assembly/EFIE.jl` | `assemble_Z_efie` |
| Excitation definition | `src/assembly/Excitation.jl` | `make_plane_wave`, `assemble_excitation` |
| Linear solve | `src/solver/Solve.jl` | `solve_forward` |
| Near-/total-field computation | `src/postprocessing/NearField.jl` | `compute_nearfield`, `compute_total_field` |
| Far‑field computation | `src/postprocessing/FarField.jl` | `radiation_vectors`, `compute_farfield` |
| Energy balance check | `src/postprocessing/Diagnostics.jl` | `energy_ratio` |
| Example script | `validation/paper/run_convergence_study.jl` | Complete convergence study |

---

## 6) Exercises

### Basic

1. **Mesh refinement**: Run the manual workflow with `Nx = Ny = 6` (instead of 4).
   Compare the condition number (`condition_diagnostics(Z).cond`) and
   the energy ratio. Does the condition number increase as expected (`∝ N³`)?
2. **Frequency sweep**: Change the frequency to 1 GHz and 10 GHz (adjust `f`).
   Compute `L/λ` each time and observe how the condition number and energy ratio
   vary with electrical size.
3. **Angular grid study**: Vary the far‑field grid resolution:
   `make_sph_grid(18,36)`, `(36,72)`, `(72,144)`. Plot the energy ratio vs.
   total number of angular points. Does it converge?

### Practical

4. **Plate size effect**: Double the plate side length to 0.2 m while keeping
   `Nx = Ny = 4`. How does the electrical size change? How does the condition
   number change? Explain.
5. **Polarization change**: Modify the incident polarization to `Vec3(0,1,0)`
   (y‑polarized). Re‑run and verify that the far‑field pattern rotates
   accordingly.
6. **Quadrature order impact**: Increase `quad_order` from 3 to 5 in both
   `assemble_Z_efie` and `radiation_vectors`. Compare the energy ratio and
   assembly time. Is the higher order worth the cost?
7. **Field-recovery check**: Evaluate `E_sca` and `E_tot` at several off-surface
   points. Verify that `E_tot - E_sca` matches the incident plane wave.

### Advanced

8. **OBJ import**: Replace the plate with an OBJ file (e.g., a simple cube).
   Repair the mesh with `repair_mesh_for_simulation`, build RWG, and run the
   same validation. Does the energy ratio stay close to 1?
9. **Memory estimation**: Write a function that, given a target maximum memory
   (e.g., 2 GiB), returns the maximum `N` allowed. Test it with the plate mesh
   by increasing `Nx`, `Ny` until the estimate exceeds your limit.
10. **Convergence study script**: Extend `validation/paper/run_convergence_study.jl` to also
    record the far‑field directivity at broadside (θ = 0°). Plot directivity vs.
    mesh refinement and observe convergence.

---

## 7) Tutorial Checklist

Before moving to the next tutorial, verify you can:

- [ ] Run the packaged convergence study (`validation/paper/run_convergence_study.jl`) and interpret its output.
- [ ] Write a minimal script that assembles and solves the EFIE for a PEC plate.
- [ ] Compute the relative residual and energy ratio, and know acceptable ranges.
- [ ] Diagnose common failures (memory, conditioning, unit mistakes) using the troubleshooting guide.
- [ ] Modify the script to change frequency, plate size, polarization, or mesh density.
- [ ] Locate the relevant source files for each step of the workflow.

---

## 8) Further Reading

- **Paper `bare_jrnl.tex`**: Section 3.1 describes the EFIE discretization and RWG basis.
- **Near-/total-field validation**: [validation/06-near-total-field-rayleigh-sphere.md](../validation/06-near-total-field-rayleigh-sphere.md) and `examples/21_near_total_field_rayleigh_sphere.jl`.
- **Classic MoM Text**: *Field Computation by Moment Methods* (Harrington, 1993) – Chapter 4 introduces the EFIE for PEC surfaces.
- **Julia for Scientific Computing**: *Julia Programming for Operations Research* (Chang, 2020) – good introduction to Julia syntax and performance.
- **Plots.jl Documentation**: [http://docs.juliaplots.org/](http://docs.juliaplots.org/) – for visualizing far‑field patterns.

---
