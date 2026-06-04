# Chapter 4: Sphere-vs-Mie Benchmark

## Purpose

Validate the complete far-field and RCS computation pipeline against an analytical reference: Mie theory for a perfect electric conductor (PEC) sphere. Unlike cross-solver validation (Bempp-cl), the sphere-Mie benchmark provides a **ground truth** with machine-precision accuracy, isolating implementation errors from solver-to-solver convention differences. This benchmark is essential for verifying radiation pattern correctness, RCS scaling, and angular dependence.

Related validation: this chapter is the exact benchmark for far-field and RCS
quantities. For analytical validation of local observation-space fields
(`compute_nearfield` and `compute_total_field`), use
[validation/06-near-total-field-rayleigh-sphere.md](06-near-total-field-rayleigh-sphere.md).

---

## Learning Goals

After this chapter, you should be able to:

1. Run the PEC sphere benchmark workflow and interpret error metrics.
2. Distinguish discretization errors from implementation bugs through mesh refinement studies.
3. Understand Mie theory implementation and its limitations for validation.
4. Set acceptance thresholds for RCS accuracy in your applications.
5. Extend the benchmark to custom sphere geometries and frequencies.

---

## 1) Theoretical Foundation

### 1.1 Mie Theory for PEC Spheres

For a sphere of radius $a$ illuminated by a plane wave at wavenumber $k$, Mie theory provides an exact analytical solution for the scattered field. The bistatic radar cross section (RCS) is:

```math
\sigma(\gamma) = \frac{\lambda^2}{\pi} \left| \sum_{n=1}^\infty \frac{2n+1}{n(n+1)} \left( a_n \pi_n(\cos\gamma) + b_n \tau_n(\cos\gamma) \right) \right|^2
```

where $\gamma$ is the scattering angle, $\pi_n$ and $\tau_n$ are angular functions, and $a_n$, $b_n$ are Mie coefficients. For PEC spheres, the coefficients simplify to ratios of spherical Bessel functions.

### 1.2 Implementation in `Mie.jl`

The package provides two primary functions:

- **`mie_s1s2_pec(x, μ)`**: Compute Mie scattering amplitudes $S_1$, $S_2$ at size parameter $x = ka$ and $\mu = \cos\gamma$
- **`mie_bistatic_rcs_pec(k, a, k_inc_hat, pol_inc, rhat)`**: Compute bistatic RCS for specific geometry and polarization

The implementation uses recurrence relations for spherical Bessel functions and Legendre functions, truncated at $n_{\text{max}} = \max(3, \lceil x + 4x^{1/3} + 2 \rceil)$ for machine precision.

### 1.3 Validation Metrics

The benchmark compares MoM-computed RCS $\sigma_{\text{MoM}}$ to Mie reference $\sigma_{\text{Mie}}$ on a $\phi = \text{constant}$ cut:

1. **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum_i |\Delta\text{dB}_i|$
2. **Root Mean Square Error (RMSE)**: $\sqrt{\frac{1}{N} \sum_i (\Delta\text{dB}_i)^2}$
3. **Maximum Absolute Error**: $\max_i |\Delta\text{dB}_i|$
4. **Backscatter Error**: $\Delta\sigma_{\text{bs}}$ at $\gamma = 180^\circ$

where $\Delta\text{dB}_i = 10\log_{10}(\sigma_{\text{MoM},i}) - 10\log_{10}(\sigma_{\text{Mie},i})$.

---

## 2) Practical Benchmark Workflow

### 2.1 Running the Benchmark

```bash
julia --project=. examples/04_pec_sphere_mie.jl
```

This example script currently runs with built-in parameters (icosphere geometry,
radius, frequency). For custom sweeps, edit `a`, `freq`, and `subdivisions`
inside `examples/04_pec_sphere_mie.jl`.

### 2.2 Reported Outputs

The script prints benchmark metrics to stdout:
- MAE, RMSE, max-absolute dB error on the selected $\phi$-cut
- Backscatter mismatch in dB
- Sample rows of $\gamma$, $\sigma_{\mathrm{MoM}}$, $\sigma_{\mathrm{Mie}}$, and $\Delta$

If you need CSV/figure artifacts, add a post-processing block that writes
`γ`, `σ_mom_cut`, `σ_mie`, and `ΔdB`.

### 2.3 Example Output

Typical results for a 0.05 m radius sphere at 3.0 GHz ($ka \approx 3.14$):

```
── MoM vs Mie summary (φ=0.0° cut) ──
  MAE(dB):  0.856
  RMSE(dB): 1.124
  Max |Δ|(dB): 3.217
  Backscatter Δ(dB): 0.543
```

**Interpretation**: 
- Mean error < 1 dB indicates good overall agreement
- Maximum error ~3 dB occurs at deep nulls where small absolute errors magnify in dB scale
- Backscatter error < 0.6 dB validates monostatic RCS computation

---

## 3) Interpreting Results and Setting Acceptance Criteria

### 3.1 Expected Error Sources

1. **Discretization error**: Finite mesh representation of curved surface
2. **Basis function limitations**: RWG functions approximate surface currents
3. **Numerical integration error**: Quadrature accuracy for curved elements
4. **Far-field approximation**: Radiation vectors computed from discretized currents

### 3.2 Mesh Refinement Study

Error should decrease with mesh refinement:

```julia
# Run benchmark at multiple mesh densities
meshes = ["sphere_coarse.obj", "sphere_medium.obj", "sphere_fine.obj"]
errors = []

for mesh in meshes
    # Run benchmark and extract MAE
    mae = run_benchmark_and_get_mae(mesh)
    push!(errors, mae)
end

# Plot convergence: error ~ O(h^p)
```

**Expected trend**: Errors decrease as element size $h \to 0$, with convergence rate $p \approx 1-2$ for RCS.

### 3.3 Acceptance Thresholds

Based on typical results:

- **Excellent agreement**: MAE < 0.5 dB, Max |Δ| < 2.0 dB
- **Good agreement**: MAE < 1.0 dB, Max |Δ| < 3.0 dB  
- **Needs investigation**: MAE > 1.5 dB, Max |Δ| > 5.0 dB
- **Likely bug**: MAE > 3.0 dB across multiple mesh densities

**Note**: Deep nulls (← -20 dB) can show large dB errors from small absolute errors. Focus on main-lobe and sidelobe regions for validation.

### 3.4 Distinguishing Bugs from Discretization Limits

1. **Global offset** (all points shifted by ~X dB): Likely normalization or scaling bug
2. **Angular shift** (pattern translated in γ): Geometry or coordinate system error
3. **Localized spikes** at specific angles: Far-field integration or singularity handling issue
4. **No convergence** with mesh refinement: Implementation bug, not discretization limit

---

## 4) Advanced Benchmarking Techniques

### 4.1 Frequency Sweep

Validate across electrical sizes:

```julia
freqs = [1e9, 2e9, 3e9, 5e9, 10e9]  # 1-10 GHz
results = []

for f in freqs
    ka = 2π * f / 299792458 * a
    println("ka = $ka")
    # Run benchmark at frequency f
    mae, max_err = run_benchmark_at_frequency(f)
    push!(results, (ka=ka, mae=mae, max_err=max_err))
end
```

**Expected**: Errors may increase near resonances (specific ka values) where fields are sensitive to discretization.

### 4.2 Multiple Polarizations and Incidence Angles

```julia
# Test different incidence directions
inc_angles = [0, 30, 60, 90]  # degrees from broadside

for θ_inc in inc_angles
    k_inc = Vec3(sin(θ_inc*π/180), 0, -cos(θ_inc*π/180))
    # Run benchmark with rotated incidence
    errors = run_benchmark(k_inc=k_inc)
end
```

**Purpose**: Verify far-field computation works correctly for oblique incidence.

### 4.3 Convergence to Analytical Solution

The ultimate test: error → 0 as mesh refines:

```julia
# Sequence of progressively refined sphere meshes
mesh_sizes = [0.1, 0.05, 0.025, 0.0125]  # target edge lengths

conv_rates = []
for h in mesh_sizes
    mesh = generate_sphere_mesh(a, h)
    error = run_benchmark(mesh)
    push!(conv_rates, (h=h, error=error))
end

# Fit log(error) ~ log(h) + constant
# Slope gives convergence rate p
```

---

## 5) Code Implementation Details

### 5.1 Mie Theory Implementation (`src/postprocessing/Mie.jl`)

Key algorithms:
- **Spherical Bessel functions**: Recurrence relations for $j_n(x)$, $y_n(x)$
- **Mie coefficients**: $a_n = -\psi_n'(x)/\xi_n'(x)$, $b_n = -\psi_n(x)/\xi_n(x)$
- **Angular functions**: Recurrence for $\pi_n(\mu)$, $\tau_n(\mu)$
- **Vector scattering**: Coordinate transformations for arbitrary incidence/observation

### 5.2 Benchmark Script (`examples/04_pec_sphere_mie.jl`)

Workflow:
1. **Mesh generation/loading**: Fallback icosphere or custom OBJ
2. **Radius estimation**: From vertex positions (handles imperfect spheres)
3. **MoM solve**: Standard EFIE assembly and solution
4. **Far-field computation**: Radiation vectors → bistatic RCS
5. **Mie reference**: Analytical RCS at same angles
6. **Error computation**: MAE, RMSE, max error, backscatter error
7. **Output generation**: CSV files and plot

### 5.3 Radius Estimation Heuristic

Since OBJ spheres may not be perfect, the benchmark script estimates the radius
from vertex positions. There is no built-in `estimate_sphere_radius` function;
the estimation is done inline in the example script:

```julia
using Statistics: mean, std
using LinearAlgebra: norm

nv = size(mesh.xyz, 2)
ctr = vec(mean(mesh.xyz, dims=2))
radii = [norm(Vec3(mesh.xyz[:, i]) - Vec3(ctr)) for i in 1:nv]
a_est = mean(radii)
a_std = std(radii)
```

**Usage**: The estimated radius `a_est` is passed to the Mie reference computation. The standard deviation `a_std` indicates mesh quality; if `a_std > 0.01 * a_est`, the mesh is significantly non-spherical.

---

## 6) Troubleshooting Common Issues

### 6.1 Large Errors (> 5 dB)

1. **Check radius estimation**:
   ```julia
   nv = size(mesh.xyz, 2)
   ctr = vec(mean(mesh.xyz, dims=2))
   radii = [norm(Vec3(mesh.xyz[:, i]) - Vec3(ctr)) for i in 1:nv]
   a_est = mean(radii)
   a_std = std(radii)
   println("Estimated radius: $a_est ± $a_std m")
   ```
   If $a_{\text{std}} > 0.01a$, mesh is non-spherical -- use better mesh.

2. **Verify frequency and units**:
   - Ensure `freq` in Hz, not GHz
   - Check `a` in meters (Mie uses SI units)
   - Confirm $k = 2\pi f/c$ correct

3. **Inspect far-field scaling**:
   ```julia
   # Compare absolute RCS values, not just dB
   println("σ_mom at γ=0: $(σ_mom_cut[1]) m²")
   println("σ_mie at γ=0: $(σ_mie[1]) m²")
   ```

### 6.2 Pattern Shift (Angular misalignment)

1. **Check coordinate systems**:
   - Mie uses $\gamma = \angle(\hat{k}_{\text{inc}}, \hat{r})$
   - MoM uses global $(\theta, \phi)$ coordinates
   - Transformation: $\gamma = \cos^{-1}(\hat{k}_{\text{inc}} \cdot \hat{r})$

2. **Verify incidence direction**:
   ```julia
   println("k_inc_hat: $k_inc_hat")
   println("Should be unit vector pointing toward sphere")
   ```

### 6.3 Poor Convergence with Mesh Refinement

1. **Mesh quality issues**:
   - Non-uniform triangles
   - Skewed elements
   - Incorrect normals (pointing inward)

2. **EFIE implementation**:
   - Singularity extraction parameters
   - Quadrature order (default: 3)
   - RWG normalization

3. **Far-field computation**:
   - Radiation vector assembly
   - Spherical grid density

---

## 7) Exercises

### 7.1 Basic Level

1. **Run benchmark with default settings**:
   - Execute `julia --project=. examples/04_pec_sphere_mie.jl`
   - Examine generated CSV files and plot
   - Verify MAE < 1.5 dB for fallback mesh

2. **Mesh density experiment**:
   - Generate spheres with different subdivision levels (1, 2, 3)
   - Run benchmark for each
   - Plot error vs. triangle count

### 7.2 Intermediate Level

3. **Frequency sweep**:
   - Modify benchmark to sweep ka from 1 to 10
   - Identify resonant regions where errors peak
   - Explain physical reason for increased sensitivity

4. **Acceptance threshold justification**:
   - Based on your results, propose CI thresholds:
     ```yaml
     sphere_benchmark:
       mae_db_max: 1.0
       max_abs_db_max: 3.0
       backscatter_db_max: 1.0
     ```
   - Justify each value with data from your runs

### 7.3 Advanced Level

5. **Extended validation suite**:
   - Implement benchmark for multiple incidence angles (0°, 30°, 60°)
   - Add polarization variation (TE, TM)
   - Create aggregated report with all results

6. **Convergence rate analysis**:
   - Generate sequence of 5+ progressively refined sphere meshes
   - Fit error = C * h^p
   - Compare p to theoretical expectations (1-2 for RCS)

---

## 8) Code Mapping

### 8.1 Primary Files

- **Mie theory**: `src/postprocessing/Mie.jl`
  - `mie_s1s2_pec`, `mie_bistatic_rcs_pec`
  - Internal: `_sph_bessel_jy_arrays`, `_mie_nmax`

- **Benchmark script**: `examples/04_pec_sphere_mie.jl`
  - Complete workflow from mesh to error metrics
  - Fallback icosphere generation
  - Radius estimation and validation

- **RCS diagnostics**: `src/postprocessing/Diagnostics.jl`
  - `bistatic_rcs`, `backscatter_rcs`

### 8.2 Supporting Scripts

- **Sphere RCS example**: `examples/04_pec_sphere_mie.jl`
- **Visualization utilities**: `src/postprocessing/Visualization.jl`

### 8.3 Output Artifacts

All outputs in `data/` directory:
- `sphere_mie_benchmark_phi_cut.csv`: Detailed angular comparison
- `sphere_mie_benchmark_summary.csv`: Aggregate error metrics
- `sphere_mie_benchmark_phi_cut.png`: Visual comparison plot

---

## 9) Chapter Checklist

Before considering sphere validation complete, ensure you can:

- [ ] Run benchmark and achieve MAE < 1.5 dB with default mesh
- [ ] Explain sources of discrepancy between MoM and Mie results
- [ ] Distinguish discretization errors from implementation bugs
- [ ] Perform mesh refinement study and observe error convergence
- [ ] Set appropriate acceptance thresholds for your application
- [ ] Troubleshoot common issues (large errors, pattern shifts, poor convergence)

---

## 10) Further Reading

- **Mie theory derivation**: Bohren & Huffman, *Absorption and Scattering of Light by Small Particles* (1983)
- **Numerical implementation**: Wiscombe, *Improved Mie scattering algorithms* (1980)
- **RCS validation methodology**: Knott et al., *Radar Cross Section* (1993)
- **Convergence analysis for MoM**: Peterson et al., *Computational Methods for Electromagnetics* (1998)
- **Spherical wave functions**: Jackson, *Classical Electrodynamics* (1999), Chapter 16
