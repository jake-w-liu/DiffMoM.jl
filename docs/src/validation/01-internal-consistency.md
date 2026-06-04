# Chapter 1: Internal Consistency

## Purpose

Establish confidence in the solver's correctness through self-consistent checks that require no external references. These diagnostics verify that the discrete EFIE formulation, far-field computation, and objective assembly are mathematically and physically consistent. For a lossless PEC structure, energy conservation must hold; for any structure, the linear system residual must be small, and quadratic-form objectives must match direct angular integration.

These checks are necessary but not sufficient for validating observation-space
fields. For analytical validation of `compute_nearfield` and
`compute_total_field`, follow this chapter with
[validation/06-near-total-field-rayleigh-sphere.md](06-near-total-field-rayleigh-sphere.md).

---

## Learning Goals

After this chapter, you should be able to:

1. Run the four core internal consistency checks on any new simulation case.
2. Interpret energy ratios, linear residuals, objective consistency, and condition diagnostics.
3. Diagnose common failure modes and identify likely bug classes.
4. Understand the physical and mathematical principles behind each check.
5. Implement custom validation workflows for specialized scenarios.

---

## 1) Physical and Mathematical Foundations

### 1.1 Energy Conservation for Lossless Structures

For a perfect electric conductor (PEC), all power extracted from the incident field must be re-radiated. The **energy ratio** quantifies this balance:

```math
\frac{P_{\text{rad}}}{P_{\text{in}}} \approx 1
```

where:

- **Radiated power** from the far-field pattern:
  ```math
  P_{\text{rad}} = \frac{1}{2\eta_0} \int_{\mathbb{S}^2} |\mathbf{E}^\infty(\hat{\mathbf{r}})|^2 \, d\Omega
  \approx \frac{1}{2\eta_0} \sum_{q=1}^{N_\Omega} w_q |\mathbf{E}^\infty(\hat{\mathbf{r}}_q)|^2
  ```

- **Input power** delivered to the structure:
  ```math
  P_{\text{in}} = -\frac{1}{2} \Re\{\mathbf{I}^\dagger \mathbf{v}\}
  ```

For an impedance sheet with surface resistance $\Re\{Z_s\} > 0$, absorbed power reduces the ratio below unity: $P_{\text{rad}}/P_{\text{in}} < 1$.

### 1.2 Linear System Residual

The discretized EFIE must satisfy the linear system with small relative residual:

```math
\frac{\|\mathbf{Z}\mathbf{I} - \mathbf{v}\|}{\|\mathbf{v}\|} \ll 1
```

This checks the accuracy of the linear solver and the consistency of $\mathbf{Z}$ and $\mathbf{v}$ assembly.

### 1.3 Objective Consistency

Quadratic-form objectives constructed via the $\mathbf{Q}$ matrix must match direct angular integration:

```math
\mathbf{I}^\dagger \mathbf{Q} \mathbf{I} \approx \sum_{q\in\mathcal{D}} w_q \left|\mathbf{p}_q^\dagger \mathbf{E}^\infty(\hat{\mathbf{r}}_q)\right|^2
```

This verifies that $\mathbf{Q}$ construction, polarization projection $\mathbf{p}_q$, and angular weights $w_q$ are mutually consistent.

### 1.4 Condition Diagnostics

The EFIE matrix condition number $\kappa(\mathbf{Z}) = \sigma_{\max}/\sigma_{\min}$ indicates numerical stability. For well-discretized problems, $\kappa$ grows as $O(h^{-2})$ to $O(h^{-3})$ with mesh refinement. Extreme values suggest discretization issues or implementation errors.

---

## 2) Implementation in `DiffMoM.jl`

### 2.1 Energy Ratio Computation

The `energy_ratio` function in `src/postprocessing/Diagnostics.jl` implements the complete check:

```julia
ratio = energy_ratio(I, v, E_ff, grid; eta0=η0)
```

Internally, it calls:
- `radiated_power(E_ff, grid)`: Computes $P_{\text{rad}}$ via spherical quadrature
- `input_power(I, v)`: Computes $P_{\text{in}}$ from MoM currents and excitation

**Expected values:**
- PEC structures: $0.98 \lesssim \text{ratio} \lesssim 1.02$ (within 2%)
- Refined meshes: $0.99 \lesssim \text{ratio} \lesssim 1.01$ (within 1%)
- Lossy impedance: $\text{ratio} < 1$, decreasing with resistance

### 2.2 Linear Residual Check

Compute directly after solving $\mathbf{Z}\mathbf{I} = \mathbf{v}$:

```julia
residual_norm = norm(Z * I - v)
excitation_norm = norm(v)
relative_residual = residual_norm / excitation_norm
```

Acceptable values: $\text{relative\_residual} < 10^{-8}$ for direct solves, $<10^{-4}$ for iterative solves.

### 2.3 Objective Consistency Verification

Use `projected_power` to compute the angular-integrated version:

```julia
P_direct = projected_power(E_ff, grid, pol_matrix; mask=mask)
P_quadratic = real(dot(I, Q * I))
relative_error = abs(P_direct - P_quadratic) / max(abs(P_direct), abs(P_quadratic))
```

Acceptable values: $\text{relative\_error} < 10^{-12}$ for double precision.

### 2.4 Condition Number Diagnostics

The `condition_diagnostics` function provides singular value information:

```julia
cond_info = condition_diagnostics(Z)
println("Condition number: $(cond_info.cond)")
println("Largest singular value: $(cond_info.sv_max)")
println("Smallest singular value: $(cond_info.sv_min)")
```

**Typical ranges:**
- Coarse meshes (2–4 elements/λ): $\kappa \sim 10^2 - 10^3$
- Moderate meshes (6–8 elements/λ): $\kappa \sim 10^3 - 10^4$
- Fine meshes (10+ elements/λ): $\kappa \sim 10^4 - 10^5$

---

## 3) Practical Workflow and Examples

### 3.1 Complete Validation Script

The convergence study driver (`validation/paper/run_convergence_study.jl`) demonstrates systematic checks across mesh refinement:

```julia
# 1. Solve EFIE
I = Z \ v

# 2. Compute far field
E_ff = compute_farfield(G_mat, I, NΩ)

# 3. Energy ratio
e_ratio = energy_ratio(I, v, E_ff, grid)

# 4. Objective consistency
P_rad = radiated_power(E_ff, grid)
P_in = input_power(I, v)
Q = build_Q(G_mat, grid, pol_mat; mask=mask)
J_quad = real(dot(I, Q * I))
J_direct = projected_power(E_ff, grid, pol_mat; mask=mask)

# 5. Condition diagnostics
cond_info = condition_diagnostics(Z)

# 6. Linear residual
residual = norm(Z * I - v) / norm(v)
```

### 3.2 Interpreting Results from the Paper

The validation in `bare_jrnl.tex` (Section IV) reports:

- **Energy balance error within 2%** across all tested meshes
- **Within 1% for $N_x \geq 3$** (finer discretizations)
- **Reciprocity satisfied** up to numerical tolerance
- **Objective consistency** verified to machine precision

These results confirm that the EFIE assembly, far-field quadrature, and excitation scaling are mutually consistent.

---

## 4) Common Failure Modes and Diagnosis

### 4.1 Diagnostic Decision Tree

1. **Large linear residual only ($>10^{-4}$)**
   - Likely cause: Linear solver tolerance too loose, or $\mathbf{Z}$/$ \mathbf{v}$ assembly error
   - Check: Direct vs. iterative solver, quadrature order, EFIE operator assembly

2. **Good residual but poor energy ratio ($|1 - P_{\text{rad}}/P_{\text{in}}| > 0.05$)**
   - Likely cause: Far-field computation or spherical quadrature error
   - Check: Radiation vector assembly, spherical grid weights, η₀ constant

3. **Good energy ratio but objective inconsistency ($>10^{-8}$)**
   - Likely cause: $\mathbf{Q}$ matrix construction or polarization projection mismatch
   - Check: `build_Q` parameters, `pol_matrix` definition, mask alignment

4. **Extreme condition number ($\kappa > 10^6$ for coarse mesh)**
   - Likely cause: Mesh quality issues or EFIE operator implementation error
   - Check: Triangle aspect ratios, RWG basis function orientation, singularity extraction

5. **Intermittent failures with geometry import**
   - Likely cause: Mesh preprocessing pipeline inconsistency
   - Check: STL import tolerances, normal orientation, duplicate vertices

### 4.2 Example: Debugging Energy Ratio Mismatch

If `energy_ratio` returns 1.15 (15% excess radiation):

1. **Verify far-field scaling**:
   ```julia
   # Check individual components
   P_rad_calc = radiated_power(E_ff, grid; eta0=η0)
   P_in_calc = input_power(I, v)
   
   # Manual recomputation for verification
   P_rad_manual = sum(grid.w .* sum(abs2, E_ff, dims=1)[:]) / (2 * η0)
   P_in_manual = -0.5 * real(dot(I, v))
   ```

2. **Check η₀ constant consistency**:
   - Ensure same value used in EFIE assembly and `radiated_power`
   - Default: `η₀ = 376.730313668` Ω (free-space impedance)

3. **Verify spherical grid normalization**:
   - Total solid angle: `sum(grid.w) ≈ 4π`
   - Acceptable tolerance: `|sum(grid.w) - 4π| < 1e-10`

---

## 5) Advanced Topics

### 5.1 Reciprocity Checks

For lossless structures, the EFIE matrix should be symmetric (reciprocal):

```julia
reciprocity_error = norm(Z - transpose(Z)) / norm(Z)
```

Expected: `reciprocity_error < 1e-10` for PEC with symmetric quadrature.

### 5.2 Passivity Verification

For impedance sheets, verify power decomposition:
- Resistive part ($\Re\{Z_s\} > 0$) contributes only to $\Re\{\mathbf{Z}\}$
- Purely reactive sheet ($\Im\{Z_s\} \neq 0$) contributes only to $\Im\{\mathbf{Z}\}$

### 5.3 Mesh Convergence Monitoring

Energy ratio should approach unity under mesh refinement:

```julia
# Expected convergence trend
for Nx in [2, 3, 4, 5, 6, 8]
    mesh = make_rect_plate(Lx, Ly, Nx, Nx)
    # ... solve and compute energy_ratio
    # ratio should approach 1.0 as Nx increases
end
```

---

## 6) Code Mapping

### 6.1 Primary Implementation Files

- **Diagnostic functions**: `src/postprocessing/Diagnostics.jl`
  - `radiated_power`, `input_power`, `energy_ratio`, `condition_diagnostics`
  - `bistatic_rcs`, `backscatter_rcs`

- **Objective assembly**: `src/optimization/QMatrix.jl`
  - `build_Q`

- **Projected power**: `src/postprocessing/Diagnostics.jl`
  - `projected_power`

- **Far-field computation**: `src/postprocessing/FarField.jl`
  - `radiation_vectors`, `compute_farfield`

- **Forward solver**: `src/solver/Solve.jl`, `src/assembly/EFIE.jl`, `src/assembly/Excitation.jl`

### 6.2 Example Scripts

- **Convergence study**: `validation/paper/run_convergence_study.jl`
- **Paper consistency aggregation**: `validation/paper/generate_consistency_report.jl`
- **Sphere benchmark**: `examples/04_pec_sphere_mie.jl`
- **Near-/total-field benchmark**: `examples/21_near_total_field_rayleigh_sphere.jl`

---

## 7) Exercises

### 7.1 Basic Level

1. **Run complete checks on a PEC plate**:
   - Load a 1λ × 1λ plate with $N_x = 4$ elements per side
   - Compute all four consistency gates
   - Verify energy ratio $0.98 < P_{\text{rad}}/P_{\text{in}} < 1.02$

2. **Debug an intentional error**:
   - Perturb spherical grid weights: `grid.w .*= 1.1`
   - Observe which checks fail and explain why

### 7.2 Intermediate Level

3. **Mesh convergence study**:
   - Reproduce the convergence study from `validation/paper/run_convergence_study.jl`
   - Plot energy ratio vs. mesh density
   - Verify condition number growth: $\kappa \sim O(h^{-p})$

4. **Impedance sheet validation**:
   - Test a purely reactive sheet ($Z_s = i200\,\Omega$)
   - Confirm energy ratio $\approx 1$ (a purely reactive sheet does not absorb power)
   - Verify reciprocity still holds

### 7.3 Advanced Level

5. **Custom diagnostic implementation**:
   - Implement a reciprocity check for impedance-loaded structures
   - Validate against analytical expectations

6. **Failure injection and diagnosis**:
   - Intentionally mis-scale excitation vector `v`
   - Document the cascade of failing checks
   - Propose debugging steps

---

## 8) Chapter Checklist

Before proceeding to external validation, ensure you can:

- [ ] Compute energy ratio for PEC and impedance cases
- [ ] Verify linear residual $< 10^{-8}$ for direct solves
- [ ] Confirm objective consistency to machine precision
- [ ] Interpret condition numbers for mesh quality assessment
- [ ] Diagnose common failure patterns
- [ ] Run the convergence study script successfully

---

## 9) Further Reading

- **EFIE theory and validation**: Peterson et al., *Computational Methods for Electromagnetics* (1998)
- **Energy conservation in MoM**: Chew et al., *Fast and Efficient Algorithms in Computational Electromagnetics* (2001)
- **Reciprocity theorems**: Harrington, *Time-Harmonic Electromagnetic Fields* (1961)
- **Numerical verification**: Roache, *Verification and Validation in Computational Science and Engineering* (1998)
- **Paper reference**: `bare_jrnl.tex`, Section IV (verification results and energy balance)
