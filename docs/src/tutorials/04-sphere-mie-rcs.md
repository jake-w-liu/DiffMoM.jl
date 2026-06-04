# Tutorial: Sphere-Mie RCS

## Purpose

Validation against analytical solutions is essential for building confidence in any computational electromagnetics code. This tutorial walks through a canonical benchmark: computing the radar cross‑section (RCS) of a perfect electric conductor (PEC) sphere using the method of moments and comparing it with the exact Mie series solution.

This tutorial is intentionally focused on far-field / RCS validation. For
analytical validation of `compute_nearfield` and `compute_total_field`, use the
companion Rayleigh benchmark in
[validation/06-near-total-field-rayleigh-sphere.md](../validation/06-near-total-field-rayleigh-sphere.md) and
`examples/21_near_total_field_rayleigh_sphere.jl`.

You will learn to:

- **Generate sphere meshes** (icosphere fallback or imported OBJ) and compute MoM RCS.
- **Compute analytical Mie RCS** using the built‑in `mie_bistatic_rcs_pec` function.
- **Compare MoM and Mie results** quantitatively with error metrics (MAE, RMSE, backscatter difference).
- **Diagnose mismatch sources** (mesh density, radius estimation, quadrature order, incident‑wave alignment).
- **Interpret convergence trends** as mesh refines, confirming that MoM converges to the analytical solution.

**Why spheres?** The PEC sphere is the only closed‑form exact solution for 3D scattering (Mie series). It provides a rigorous validation test for EFIE implementations, singular‑integration accuracy, and far‑field computation.

---

## Learning Goals

After this tutorial, you should be able to:

1. **Understand the Mie series** for PEC sphere scattering and its implementation in `src/postprocessing/Mie.jl`.
2. **Generate sphere meshes** with controllable resolution (icosphere subdivisions or imported OBJ).
3. **Run the end‑to‑end MoM workflow** for bistatic RCS: EFIE assembly, plane‑wave excitation, far‑field computation.
4. **Compute error metrics** (MAE, RMSE, backscatter Δ) between MoM and Mie results.
5. **Diagnose common discrepancies** and distinguish mesh‑convergence issues from implementation errors.
6. **Refine the mesh** and observe error reduction, confirming correct asymptotic convergence.

---

## Mathematical Background

### Mie Series for PEC Sphere

For a sphere of radius $a$ illuminated by a plane wave $\mathbf{E}^\text{inc} = \mathbf{p}_0 e^{-i\mathbf{k}\cdot\mathbf{r}}$, the scattered field can be expanded in vector spherical harmonics. The bistatic RCS $\sigma(\hat{\mathbf{r}})$ (radar cross‑section per unit solid angle) is given by the Mie series (paper Eq. 40):

\[
\sigma(\hat{\mathbf{r}}) = \frac{\lambda^2}{4\pi} \left| \sum_{n=1}^\infty \frac{(-i)^n (2n+1)}{n(n+1)} \left[ a_n \mathbf{X}_{n1}(\hat{\mathbf{r}}) + b_n \mathbf{X}_{n2}(\hat{\mathbf{r}}) \right] \right|^2,
\]

where $a_n, b_n$ are the Mie coefficients for a PEC sphere:

\[
a_n = -\frac{j_n(ka)}{h_n^{(1)}(ka)}, \qquad
b_n = -\frac{[ka\,j_n(ka)]'}{[ka\,h_n^{(1)}(ka)]'}.
\]

The function `mie_bistatic_rcs_pec(k, a, khat_inc, pol, rhat)` (`src/postprocessing/Mie.jl:3`) computes this series truncated at $n_\text{max} = \max(3,\, \lceil x + 4\,x^{1/3} + 2 \rceil)$ where $x = ka$, providing machine‑precision reference values.

### EFIE Formulation for Sphere

The electric‑field integral equation (EFIE) for a PEC sphere is identical to the general EFIE (paper Eq. 5). The sphere’s smooth curvature and absence of sharp edges make it a benign test case—singular integrals dominate but are handled by the same `analytical_integral_1overR` routine used for general geometries.

### Convergence Metric

Define the **mean absolute error** (MAE) in dB scale over a cut of $N_\theta$ directions:

\[
\text{MAE}_{\text{dB}} = \frac{1}{N_\theta} \sum_{i=1}^{N_\theta} \bigl| 10\log_{10}\sigma_{\text{MoM}}(\theta_i) - 10\log_{10}\sigma_{\text{Mie}}(\theta_i) \bigr|.
\]

For a well‑implemented MoM, $\text{MAE}_{\text{dB}}$ should decrease as mesh resolution increases (typical values: 1–3 dB for coarse meshes, 0.1–0.5 dB for fine meshes at $ka \approx 10$).

---

## Step‑by‑Step Workflow

### 1) Choose Frequency and Sphere Size

A standard benchmark uses $ka = 10$ (sphere circumference ≈ 10λ), which balances computational cost and richness of the scattering pattern. At 2 GHz ($\lambda = 15$ cm), this corresponds to radius $a = 10\lambda/(2\pi) \approx 0.24$ m.

```julia
using DiffMoM
using LinearAlgebra
using Statistics

freq = 2e9                     # 2 GHz
c0   = 299792458.0
λ0   = c0 / freq
k    = 2π / λ0
η0   = 376.730313668

# Sphere radius for ka = 10
ka_target = 10.0
a = ka_target / k
println("Frequency: $(freq/1e9) GHz,  λ = $(round(λ0*100, digits=2)) cm")
println("Target ka = $ka_target → radius a = $(round(a*100, digits=2)) cm")
```

### 2) Generate or Import a Sphere Mesh

The packaged benchmark script `examples/04_pec_sphere_mie.jl` includes a local
fallback icosphere generator (subdivided icosahedron). For manual workflows, you
can either reuse that helper or supply an externally generated OBJ file.

```julia
# For a manual workflow, provide an OBJ sphere mesh. The packaged example
# includes a local fallback icosphere helper if you want an all-in-repo path.
mesh_path = "sphere_ka10.obj"   # replace with your sphere mesh file
mesh = read_obj_mesh(mesh_path)
rwg = build_rwg(mesh)

println("Mesh: $(nvertices(mesh)) vertices, $(ntriangles(mesh)) triangles, $(rwg.nedges) RWG edges")
```

**Mesh resolution guideline:** Each triangle should be $\lesssim \lambda/5$ linear dimension for acceptable accuracy. The icosphere with `subdivisions=2` yields ~320 triangles, roughly $\lambda/3$ at $ka=10$.

### 3) Estimate Actual Radius from Mesh

Meshing imperfections may slightly alter the effective radius. Compute the mean distance from vertices to the centroid.

```julia
function estimate_sphere_radius(mesh)
    ctr = vec(mean(mesh.xyz, dims=2))
    radii = [norm(Vec3(mesh.xyz[:, i]) - Vec3(ctr)) for i in 1:nvertices(mesh)]
    return mean(radii), std(radii), Vec3(ctr)
end

a_est, a_std, ctr = estimate_sphere_radius(mesh)
println("Estimated radius: $(round(a_est, digits=6)) m  (std = $(round(a_std, digits=6)) m)")
println("Estimated center: ($(ctr[1]), $(ctr[2]), $(ctr[3]))")
```

Use `a_est` for Mie comparisons; a large standard deviation indicates a non‑spherical mesh (possible import errors).

### 4) Assemble EFIE and Solve

Standard plane‑wave excitation (x‑polarized, propagating –z).

```julia
Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=η0)

k_vec = Vec3(0.0, 0.0, -k)
E0    = 1.0
pol   = Vec3(1.0, 0.0, 0.0)   # x‑polarized
pw = make_plane_wave(k_vec, E0, pol)
v = assemble_excitation(mesh, rwg, pw; quad_order=3)

I = solve_forward(Z, v)
residual = norm(Z * I - v) / max(norm(v), 1e-30)
println("Forward solve residual: $residual")
```

### 5) Compute Bistatic RCS on a Spherical Grid

Use a dense angular grid to capture the full scattering pattern. The `bistatic_rcs` function returns RCS values for each direction.

```julia
grid = make_sph_grid(181, 72)          # 1° θ resolution, 5° φ resolution
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=η0)
E_ff = compute_farfield(G_mat, I, length(grid.w))
σ_mom = bistatic_rcs(E_ff; E0=1.0)     # σ in m² for each grid direction
```

### 6) Compute Mie Reference for the Same Directions

Select a constant‑φ cut (e.g., φ = 0°) for comparison. Compute Mie RCS for each direction in the cut.

```julia
khat_inc = k_vec / norm(k_vec)
φ_target = grid.phi[argmin(grid.phi)]   # smallest φ (≈0°)
φ_cut_idx = [q for q in eachindex(grid.w) if abs(grid.phi[q] - φ_target) < 1e-12]

# Sort by θ for a clean curve
perm = sortperm(grid.theta[φ_cut_idx])
cut_idx = φ_cut_idx[perm]

θ_cut = grid.theta[cut_idx]
σ_mie = zeros(Float64, length(cut_idx))
for (i, q) in enumerate(cut_idx)
    rhat = Vec3(grid.rhat[:, q])
    σ_mie[i] = mie_bistatic_rcs_pec(k, a_est, khat_inc, pol, rhat)
end

σ_mom_cut = σ_mom[cut_idx]
```

### 7) Compute Error Metrics

Compare in dB scale (common for RCS).

```julia
dB_mom = 10 .* log10.(max.(σ_mom_cut, 1e-30))
dB_mie = 10 .* log10.(max.(σ_mie, 1e-30))
ΔdB = dB_mom .- dB_mie

mae_db = mean(abs.(ΔdB))
rmse_db = sqrt(mean(abs2.(ΔdB)))
max_abs_db = maximum(abs.(ΔdB))

# Backscatter (θ = 180°) comparison
σ_bs_mom = backscatter_rcs(E_ff, grid, khat_inc; E0=1.0).sigma
σ_bs_mie = mie_bistatic_rcs_pec(k, a_est, khat_inc, pol, -khat_inc)
Δbs_db = 10 * log10(max(σ_bs_mom, 1e-30)) - 10 * log10(max(σ_bs_mie, 1e-30))

println("Error metrics (φ = $(round(rad2deg(φ_target), digits=2))° cut):")
println("  MAE(dB)    = $mae_db")
println("  RMSE(dB)   = $rmse_db")
println("  Max |Δ|(dB)= $max_abs_db")
println("  Backscatter Δ(dB) = $Δbs_db")
```

### 8) Plot Comparison

```julia
using Plots

γ = acos.(clamp.(dot.(Ref(khat_inc), [Vec3(grid.rhat[:, q]) for q in cut_idx]), -1.0, 1.0))

p = plot(rad2deg.(γ), dB_mom;
    lw=2, label="MoM",
    xlabel="Scattering angle γ (deg)",
    ylabel="Bistatic RCS (dBsm)",
    title="PEC sphere, ka = $(round(k*a_est, digits=2))")
plot!(p, rad2deg.(γ), dB_mie; lw=2, ls=:dash, label="Mie (exact)")
savefig("sphere_mie_comparison.png")
```

### 9) Run the Full Benchmark Script

For a complete benchmark that automates all steps and writes CSV outputs, use the provided example:

```bash
julia --project=. examples/04_pec_sphere_mie.jl
```

Optional arguments: provide an external OBJ mesh and/or frequency in GHz.

```bash
julia --project=. examples/04_pec_sphere_mie.jl
```

Edit `a`, `freq`, and `subdivisions` inside the script for custom studies.

---

## Interpretation Guidelines

### Expected Error Magnitudes

| Mesh resolution (triangles per λ²) | Typical MAE (dB) | Comments |
|-----------------------------------|------------------|----------|
| < 10 (very coarse)                | 3–10 dB          | Large amplitude errors, pattern distorted |
| 10–30 (moderate)                  | 1–3 dB           | Shape roughly correct, nulls shallow |
| 30–100 (fine)                     | 0.3–1 dB         | Good agreement, nulls start to appear |
| > 100 (very fine)                 | < 0.3 dB         | Excellent agreement, limited by quadrature order |

For $ka = 10$, a mesh with ~1000 triangles typically yields MAE ≈ 0.5 dB.

### Common Discrepancy Patterns

| Pattern | Likely Cause | Action |
|---------|--------------|--------|
| **Constant offset** (MoM systematically higher/lower) | Incorrect radius estimation, wrong `E0` scaling | Re‑estimate `a_est`, verify `bistatic_rcs(E0=1.0)` |
| **Shape mismatch at wide angles** | Insufficient mesh density for creeping‑wave modes | Increase subdivisions, use `subdivisions=3` or higher |
| **Spikes at nulls** | Numerical noise amplified by dB scale; MoM nulls shallower than Mie | Accept as normal; compare linear scale near nulls |
| **Asymmetry in φ cuts** | Mesh not centered at origin, incident‑wave direction misaligned | Center mesh (`mesh.xyz .-= ctr`), verify `khat_inc` |
| **Large backscatter error** | Singular‑integration inaccuracies for self‑terms | Increase `quad_order` (e.g., 5) for EFIE assembly |

### Convergence Testing

To verify implementation correctness, repeat with increasing mesh density and observe error reduction. The error should decay as $h^p$ where $h$ is the mean triangle edge length and $p ≈ 1$–2 (depending on quadrature order). Use the script `examples/04_pec_sphere_mie.jl` with different `subdivisions` values.

---

## Troubleshooting

### Error 1: Mie RCS Returns NaN or Zero

- **Cause:** `ka` too large (> 150) causes overflow in spherical Bessel functions.
- **Fix:** Reduce frequency or use a smaller sphere. The Mie implementation is stable for $ka \le 150$.

### Error 2: MoM RCS Much Smaller Than Mie (e.g., –30 dB)

- **Cause:** Incorrect `E0` scaling in `bistatic_rcs`. The default `E0=1.0` assumes unit‑amplitude incident field.
- **Fix:** Ensure `assemble_v_plane_wave` uses the same `E0` (default 1.0). Pass `E0=1.0` to `bistatic_rcs`.

### Error 3: Large Standard Deviation in Radius Estimate

- **Cause:** Mesh is not spherical (imported OBJ may have artifacts).
- **Fix:** Use `mesh_quality_report(mesh)` to check for degenerate triangles. Repair with `repair_mesh_for_simulation`.

### Error 4: Residual Large (> 1e‑5)

- **Cause:** Ill‑conditioned $Z$ matrix for dense meshes.
- **Fix:** Enable preconditioning (`make_left_preconditioner`) or add small regularization (`regularization_alpha=1e‑8`).

### Error 5: Comparison Cut Shows No Signal

- **Cause:** φ‑cut selection picks a plane where scattering is minimal (e.g., φ = 90° for x‑polarized incidence).
- **Fix:** Use `φ_target = 0.0` (incidence plane) where scattering is strongest.

---

## Code Mapping

| Task | Function | Source File | Key Lines |
|------|----------|-------------|-----------|
| **Mie RCS** | `mie_bistatic_rcs_pec(k, a, khat_inc, pol, rhat)` | `src/postprocessing/Mie.jl` | 30–80 |
| **Sphere mesh generation** | `write_obj_mesh` (internal) | `examples/04_pec_sphere_mie.jl` | 27–83 |
| **Radius estimation** | `estimate_sphere_radius` (internal) | `examples/04_pec_sphere_mie.jl` | 85–89 |
| **Bistatic RCS** | `bistatic_rcs(E_ff; E0)` | `src/postprocessing/Diagnostics.jl` | 60–80 |
| **Backscatter RCS** | `backscatter_rcs(E_ff, grid, khat_inc; E0)` | `src/postprocessing/Diagnostics.jl` | 100–120 |
| **Far‑field computation** | `compute_farfield(G_mat, I, NΩ)` | `src/postprocessing/FarField.jl` | 200–220 |
| **Complete benchmark** | `04_pec_sphere_mie.jl` | `examples/` | full script |

**Standalone MoM sphere RCS** (no Mie comparison): `examples/04_pec_sphere_mie.jl`.

---

## Exercises

### Basic (45 minutes)

1. **Run the benchmark** with the default icosphere (`subdivisions=2`). Record MAE, RMSE, and backscatter error.
2. **Plot the φ = 0° cut** (MoM vs Mie) and identify angles where discrepancy is largest.
3. **Estimate the effective radius** of the mesh and compare with the nominal radius $a = ka/k$.

### Practical (90 minutes)

1. **Mesh convergence study**: Run the benchmark with `subdivisions = 1, 2, 3, 4`. Plot MAE vs number of triangles (or vs mean edge length / λ). Confirm error decreases with refinement.
2. **Quadrature order sensitivity**: Repeat with `quad_order = 2, 3, 5` in `assemble_Z_efie`. How does accuracy change? Is the default `quad_order=3` sufficient?
3. **Polarization rotation**: Change incident polarization to `pol = Vec3(0.0, 1.0, 0.0)` (y‑polarized). Compare the φ = 90° cut; explain the symmetry change.

### Advanced (2 hours)

1. **OBJ import validation**: Download a sphere OBJ from a CAD tool (e.g., ico‑sphere from Blender). Run the benchmark and compare errors with the built‑in icosphere. Does the imported mesh produce similar accuracy?
2. **Frequency sweep**: Fix sphere radius $a = 0.1$ m and sweep frequency from 1 GHz to 10 GHz ($ka = 4.2$ to $42$). Plot MAE vs $ka$ and identify the regime where MoM accuracy degrades.
3. **Extend to dielectric sphere**: Modify the Mie function to handle dielectric spheres (use `mie_s1s2_pec` as starting point). Compare with a MoM simulation using impedance boundary condition (IBC) approximation.

---

## Tutorial Checklist

Before declaring your MoM implementation validated, complete these steps:

- [ ] **Run the benchmark** with default settings and achieve MAE < 2 dB.
- [ ] **Verify mesh convergence**: errors decrease with finer meshing.
- [ ] **Check backscatter agreement**: Δ < 1 dB.
- [ ] **Inspect null locations**: MoM nulls within 5° of Mie nulls.
- [ ] **Document results**: save comparison plot and error metrics for your records.

---

## Further Reading

- **Paper Section 5.1** – Sphere benchmark as a validation gate.
- **Companion local-field benchmark**: [validation/06-near-total-field-rayleigh-sphere.md](../validation/06-near-total-field-rayleigh-sphere.md) and `examples/21_near_total_field_rayleigh_sphere.jl`.
- **`src/postprocessing/Mie.jl`** – Implementation of Mie series for PEC sphere.
- **`src/postprocessing/Diagnostics.jl`** – RCS computation utilities.
- **Tutorial 5: Airplane RCS** – Applying the validated MoM to a complex platform.
- **Balanis, *Advanced Engineering Electromagnetics*, Chapter 11** – Detailed derivation of Mie series.
