# Chapter 2: Forward Pipeline

## Purpose

Provide a complete mathematical and computational description of the forward scattering solution pipeline: from geometry and excitation definition to surface current solution, then onward to field postprocessing. This chapter covers EFIE operator assembly, impedance loading, numerical solution strategies, preconditioning, and validation diagnostics. Understanding this pipeline is essential for both forward analysis and gradient-based optimization.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive the discretized EFIE formulation for PEC and impedance boundary conditions.
2. Assemble MoM matrices with proper singularity handling and quadrature.
3. Incorporate surface impedance contributions via patch-wise mass matrices.
4. Select appropriate solution strategies (direct vs. iterative) based on problem characteristics.
5. Apply preconditioning to improve numerical conditioning.
6. Postprocess solved currents into scattered near fields, total electric fields, and far-field observables.
7. Validate forward solutions through residual checks, energy conservation, and condition diagnostics.
8. Troubleshoot common assembly and solution errors.

---

## 1) Mathematical Formulation

### 1.1 Electric Field Integral Equation (EFIE)

For a perfect electric conductor (PEC) illuminated by incident field $\mathbf{E}^{\text{inc}}$, the surface current $\mathbf{J}$ satisfies:

```math
\hat{\mathbf{n}} \times \mathbf{E}^{\text{inc}}(\mathbf{r}) = \hat{\mathbf{n}} \times i\omega\mu_0 \int_\Gamma G(\mathbf{r},\mathbf{r}')\mathbf{J}(\mathbf{r}')d\Gamma' \quad \mathbf{r} \in \Gamma
```

where $G(\mathbf{r},\mathbf{r}') = e^{-ikR}/(4\pi R)$ is the free-space Green's function with $R = |\mathbf{r}-\mathbf{r}'|$, $k = \omega\sqrt{\mu_0\epsilon_0}$ is the wavenumber, and $\eta_0 = \sqrt{\mu_0/\epsilon_0}$ is the free-space impedance.

### 1.2 Impedance Boundary Condition (IBC)

For surfaces with impedance $Z_s(\mathbf{r})$, the boundary condition becomes:

```math
\hat{\mathbf{n}} \times \mathbf{E}^{\text{inc}}(\mathbf{r}) = \hat{\mathbf{n}} \times \left[ i\omega\mu_0 \int_\Gamma G(\mathbf{r},\mathbf{r}')\mathbf{J}(\mathbf{r}')d\Gamma' + Z_s(\mathbf{r})\mathbf{J}(\mathbf{r}) \right]
```

The impedance term contributes a local operator proportional to the surface identity.

### 1.3 RWG Discretization

Expand surface current in RWG basis functions $\mathbf{f}_m(\mathbf{r})$:

```math
\mathbf{J}(\mathbf{r}) \approx \sum_{m=1}^N I_m \mathbf{f}_m(\mathbf{r})
```

Applying Galerkin testing with functions $\mathbf{f}_n(\mathbf{r})$ yields the linear system:

```math
\sum_{m=1}^N Z_{nm} I_m = v_n
```

where:

```math
Z_{nm} = -i\omega\mu_0 \left[ \int_\Gamma \int_\Gamma \mathbf{f}_n(\mathbf{r}) \cdot \mathbf{f}_m(\mathbf{r}') \, G(\mathbf{r},\mathbf{r}') \, d\Gamma \, d\Gamma' - \frac{1}{k^2} \int_\Gamma \int_\Gamma (\nabla \cdot \mathbf{f}_n)(\nabla' \cdot \mathbf{f}_m) \, G(\mathbf{r},\mathbf{r}') \, d\Gamma \, d\Gamma' \right] - \int_\Gamma Z_s(\mathbf{r}) \mathbf{f}_n(\mathbf{r}) \cdot \mathbf{f}_m(\mathbf{r}) d\Gamma
```

```math
v_n = -\int_\Gamma \mathbf{f}_n(\mathbf{r}) \cdot \mathbf{E}^{\text{inc}}(\mathbf{r}) d\Gamma
```

### 1.4 Matrix Structure

The EFIE matrix $\mathbf{Z}$ is complex symmetric (for reciprocal media) and dense. Its condition number grows as $O(h^{-2})$ to $O(h^{-3})$ with mesh refinement $h$, making iterative solution challenging without preconditioning.

---

## 2) Implementation in `DiffMoM.jl`

### 2.1 Pipeline Overview

The forward solution follows this sequence:

```math
\text{Mesh} \rightarrow \text{RWG basis} \rightarrow \text{EFIE matrix} \rightarrow \text{Impedance terms} \rightarrow \text{Excitation} \rightarrow \text{Solve} \rightarrow \text{Field Postprocessing} \rightarrow \text{Diagnostics}
```

### 2.2 Core Functions and Data Flow

```julia
# 1. Geometry and basis
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg = build_rwg(mesh; precheck=true)

# 2. EFIE operator assembly
Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=η0)

# 3. Impedance contribution
partition = PatchPartition(patch_ids, num_patches)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
Z_full = assemble_full_Z(Z_efie, Mp, theta; reactive=true)

# 4. Excitation
pw = make_plane_wave(k_vec, E0, pol_inc)
v = assemble_excitation(mesh, rwg, pw; quad_order=3)

# 5. Solution
I = solve_forward(Z_full, v)

# 6. Field postprocessing
obs = [Vec3(0.0, 0.0, 0.15)]
E_sca = compute_nearfield(mesh, rwg, I, obs, k; quad_order=3, eta0=η0)
E_tot = compute_total_field(mesh, rwg, I, pw, obs, k; quad_order=3, eta0=η0)

# 7. Diagnostics
cond_info = condition_diagnostics(Z_full)
residual = norm(Z_full * I - v) / norm(v)
```

### 2.3 EFIE Matrix Assembly (`src/assembly/EFIE.jl`)

The `assemble_Z_efie` function implements singularity-extracted quadrature:

```julia
function assemble_Z_efie(mesh::TriMesh, rwg::RWGData, k::Float64;
                        quad_order::Int=3, eta0::Float64=376.730313668)
```

**Key features:**
- **Singularity extraction**: Separate treatment of $1/R$ singularity using analytical integration
- **Adaptive quadrature**: Gauss-Legendre quadrature for well-separated elements
- **Vectorization**: Block assembly for performance
- **Full assembly**: The code loops over all (m,n) pairs for generality

**Quadrature strategy:**
- **Self-term** ($m=n$): Semi-analytical integration with singularity extraction
- **Near-term** ($R < \lambda/10$): Higher-order adaptive quadrature
- **Far-term**: Standard Gauss-Legendre with reduced order

### 2.4 Impedance Loading (`src/assembly/Impedance.jl`)

Surface impedance contributes a local mass matrix:

```math
M^{(p)}_{nm} = \int_{\Gamma_p} \mathbf{f}_n(\mathbf{r}) \cdot \mathbf{f}_m(\mathbf{r}) d\Gamma
```

where $\Gamma_p$ is the $p$-th impedance patch. For patch-wise constant impedance, the full system matrix is:

```math
\mathbf{Z}_{\text{total}} = \mathbf{Z}_{\text{EFIE}} - \sum_{p=1}^P c_p \mathbf{M}^{(p)}
```

where $c_p = \theta_p$ for resistive loading (`reactive=false`) or $c_p = i\theta_p$ for reactive loading (`reactive=true`).

**Implementation:**

```julia
# Precompute patch mass matrices (geometry-dependent, parameter-independent)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

# Assemble full impedance-loaded matrix
Z_full = assemble_full_Z(Z_efie, Mp, theta; reactive=true)
```

**Parameter types:**
- **Resistive**: `theta[p]` real → $Z_s = \theta_p$
- **Reactive**: `reactive=true` → $Z_s = i\theta_p$
- **Complex**: Combine real and imaginary parts

### 2.5 Excitation Assembly (`src/assembly/Excitation.jl`)

Plane wave excitation with propagation vector $\mathbf{k}$ and polarization $\mathbf{p}$:

```math
\mathbf{E}^{\text{inc}}(\mathbf{r}) = \mathbf{p} e^{-i\mathbf{k}\cdot\mathbf{r}}
```

The right-hand side vector elements:

```math
v_n = \int_\Gamma \mathbf{f}_n(\mathbf{r}) \cdot \mathbf{p} e^{-i\mathbf{k}\cdot\mathbf{r}} d\Gamma
```

**Implementation:**

```julia
pw = make_plane_wave(k_vec, E0, pol_inc)
v = assemble_excitation(mesh, rwg, pw; quad_order=3)
```

**Supported excitation types (8 total):**
- `PlaneWaveExcitation` — plane waves (arbitrary direction, polarization)
- `PortExcitation` — port-based excitation
- `DeltaGapExcitation` — delta gap voltage source
- `DipoleExcitation` — Hertzian dipole source
- `LoopExcitation` — magnetic loop source
- `ImportedExcitation` — externally computed field data
- `PatternFeedExcitation` — feed defined by a radiation pattern
- `MultiExcitation` — combination of multiple excitation types

### 2.6 Solution Strategies (`src/solver/Solve.jl`)

#### Direct Solution (Default)
```julia
I = solve_forward(Z, v)  # Uses LU factorization (LAPACK)
```

**Advantages:**
- Robust for ill-conditioned systems
- Reusable factorization for multiple RHS
- Exact solution (to machine precision)

**Limitations:**
- $O(N^3)$ time complexity
- $O(N^2)$ memory for factors

#### Iterative Solution (GMRES)
GMRES is available via `solver=:gmres`, with `src/solver/IterativeSolve.jl` providing
the Krylov.jl wrapper. Both left and right preconditioning are supported.
For automatic method selection based on problem size, use `solve_scattering`
in `src/Workflow.jl`, which selects dense direct, dense GMRES, or ACA GMRES
depending on the number of unknowns.

After any solve, the current vector can be postprocessed into local or
asymptotic observables:

```julia
pw = make_plane_wave(k_vec, E0, pol_inc)
obs = [Vec3(0.0, 0.0, 0.15), Vec3(0.02, 0.0, 0.18)]
E_sca = compute_nearfield(mesh, rwg, I, obs, k; quad_order=3, eta0=η0)
E_tot = compute_total_field(mesh, rwg, I, pw, obs, k; quad_order=3, eta0=η0)

grid = make_sph_grid(16, 32)
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=η0)
E_ff = compute_farfield(G_mat, I, length(grid.w))
```

This is the recommended post-solve workflow for observation-space fields:
`compute_nearfield` for scattered field only, `compute_total_field` for
`E_inc + E_sca`, and `compute_farfield` for radiation and RCS quantities.

### 2.7 Preconditioning and Conditioning

The EFIE operator becomes increasingly ill-conditioned with mesh refinement. The package provides preconditioning options:

```julia
# Automatic preconditioner selection
M_eff, enabled, reason = select_preconditioner(Mp;
    mode=:off,
    iterative_solver=false)

# Prepare conditioned system
Z_cond, v_cond, fac = prepare_conditioned_system(Z_full, v;
    preconditioner_M=M_eff)

# Solve conditioned system
I_cond = solve_system(Z_cond, v_cond)
I = I_cond
```

**Preconditioner types:**
- **Mass matrix preconditioner**: $\mathbf{M}^{-1/2}$ diagonal scaling
- **Operator splitting**: Calderón preconditioner (future)
- **Domain decomposition**: For large problems

### 2.8 Validation Diagnostics

After solving, perform essential checks:

```julia
# 1. Linear system residual
residual = norm(Z_full * I - v) / norm(v)
println("Relative residual: $residual")

# 2. Condition number diagnostics
cond_info = condition_diagnostics(Z_full)
println("Condition number: $(cond_info.cond)")

# 3. Energy conservation (requires far-field)
grid = make_sph_grid(16, 32)
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=η0)
E_ff = compute_farfield(G_mat, I, length(grid.w))
energy_ratio_val = energy_ratio(I, v, E_ff, grid; eta0=η0)
println("Energy ratio (P_rad/P_in): $energy_ratio_val")
```

**Acceptance criteria:**
- **Residual**: $< 10^{-8}$ (direct), $< 10^{-4}$ (iterative)
- **Energy ratio**: $0.98 < P_{\text{rad}}/P_{\text{in}} < 1.02$ for PEC
- **Condition number**: Monitor growth with refinement

---

## 3) Practical Workflow Examples

### 3.1 Complete PEC Forward Solve

```julia
using DiffMoM

# Geometry
Lx, Ly = 0.1, 0.1  # 10 cm plate (≈1λ at 3 GHz)
mesh = make_rect_plate(Lx, Ly, 6, 6)
rwg = build_rwg(mesh; precheck=true)

# Frequency parameters
f = 3e9  # 3 GHz
c0 = 299792458.0
λ = c0 / f
k = 2π / λ
η0 = 376.730313668

# EFIE assembly
Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=η0)

# Plane wave excitation (normal incidence, x-polarized)
k_vec = Vec3(0.0, 0.0, -k)  # Propagating in -z direction
E0 = 1.0
pol_inc = Vec3(1.0, 0.0, 0.0)
pw = make_plane_wave(k_vec, E0, pol_inc)
v = assemble_excitation(mesh, rwg, pw; quad_order=3)

# Solve
I = solve_forward(Z, v)

# Optional local field samples
obs = [Vec3(0.0, 0.0, 0.15)]
E_sca = compute_nearfield(mesh, rwg, I, obs, k; quad_order=3, eta0=η0)
E_tot = compute_total_field(mesh, rwg, I, pw, obs, k; quad_order=3, eta0=η0)

# Diagnostics
println("RWG unknowns: $(rwg.nedges)")
println("Condition number: $(condition_diagnostics(Z).cond)")
println("Residual: $(norm(Z*I - v)/norm(v))")
```

### 3.2 Impedance-Loaded Surface

```julia
# Same geometry and excitation as above

# Define impedance patches (one patch per triangle for simplicity)
Nt = ntriangles(mesh)
partition = PatchPartition(collect(1:Nt), Nt)

# Precompute patch mass matrices (geometry-only)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

# Impedance values (purely reactive, 100Ω)
theta = fill(100.0, Nt)  # Z_s = i*100 Ω

# Assemble full impedance-loaded matrix
Z_full = assemble_full_Z(Z, Mp, theta; reactive=true)

# Solve
I_imp = solve_forward(Z_full, v)

# Compare currents
println("PEC current norm: $(norm(I))")
println("Impedance current norm: $(norm(I_imp))")
```

### 3.3 Conditioned Solve for Ill-Conditioned Problems

```julia
# For fine meshes or high-contrast impedance
mesh_fine = make_rect_plate(Lx, Ly, 12, 12)
rwg_fine = build_rwg(mesh_fine)
Z_fine = assemble_Z_efie(mesh_fine, rwg_fine, k; quad_order=3, eta0=η0)

# Automatic preconditioner selection
M_eff, enabled, reason = select_preconditioner(Mp; mode=:auto, iterative_solver=false)

if enabled
    println("Preconditioner enabled: $reason")
    Z_cond, v_cond, fac = prepare_conditioned_system(Z_fine, v; preconditioner_M=M_eff)
    I = solve_system(Z_cond, v_cond)
else
    Z_cond = Z_fine
    I = solve_forward(Z_fine, v)
end

# Verify preconditioner effectiveness
cond_orig = condition_diagnostics(Z_fine).cond
cond_prec = condition_diagnostics(Z_cond).cond
println("Condition number improvement: $(cond_orig/cond_prec)x")
```

### 3.4 Batch Processing Multiple Frequencies

```julia
function frequency_sweep(mesh, rwg, freqs, theta=nothing)
    results = []
    for f in freqs
        k = 2π * f / 299792458.0
        Z = assemble_Z_efie(mesh, rwg, k; quad_order=3)
        
        if theta !== nothing
            Z_full = assemble_full_Z(Z, Mp, theta; reactive=true)
        else
            Z_full = Z
        end
        
        v = assemble_v_plane_wave(mesh, rwg, Vec3(0,0,-k), 1.0, Vec3(1,0,0))
        I = solve_forward(Z_full, v)
        
        # Store results
        push!(results, (f=f, I=I, Z=Z_full))
    end
    return results
end
```

---

## 4) Troubleshooting Common Issues

### 4.1 Diagnostic Decision Tree

1. **Large residual ($> 10^{-4}$)**
   - Check mesh quality: `mesh_quality_report`
   - Verify quadrature order (increase `quad_order`)
   - Confirm frequency/geometry units (meters vs. wavelengths)

2. **Poor energy conservation ($|1 - P_{\text{rad}}/P_{\text{in}}| > 0.05$)**
   - Verify far-field grid density
   - Check $\eta_0$ consistency in EFIE and far-field
   - Inspect excitation scaling

3. **Excessive condition number ($\kappa > 10^6$)**
   - Enable preconditioning: `select_preconditioner`
   - Coarsen mesh or increase element size
   - Consider frequency scaling (low-frequency breakdown)

4. **Memory exhaustion**
   - Estimate memory: `estimate_dense_matrix_gib`
   - Use iterative solver with preconditioner
   - Coarsen mesh or use domain decomposition

5. **Impedance solution diverges**
   - Check impedance sign convention (reactive vs. resistive)
   - Verify patch partitioning matches parameter vector
   - Monitor condition number with impedance loading

### 4.2 Performance Optimization

- **Matrix assembly**: Reuse EFIE matrix for multiple frequencies (scaling)
- **Parameter studies**: Precompute geometry-dependent terms ($\mathbf{M}^{(p)}$)
- **Batch solving**: Multiple RHS with single factorization
- **Memory mapping**: Out-of-core storage for very large matrices

### 4.3 Validation Protocol

Establish a validation protocol for new geometries:

1. **Mesh validation**: `mesh_quality_report`, `assert_mesh_quality`
2. **Basis validation**: `build_rwg` with `precheck=true`
3. **Operator validation**: Reciprocal check $\|\mathbf{Z} - \mathbf{Z}^T\|/\|\mathbf{Z}\|$
4. **Solution validation**: Residual, energy conservation, condition number
5. **Convergence check**: Refine mesh, monitor solution convergence

---

## 5) Advanced Topics

### 5.1 Multi-Frequency and Broadband Analysis

For broadband response, consider:

- **Frequency interpolation**: Solve at selected frequencies, interpolate
- **Model order reduction**: Pade approximation via moment matching
- **Asymptotic waveform evaluation**: Taylor expansion about center frequency

### 5.2 Parallel and Distributed Computing

- **Thread-parallel assembly**: BLAS-level parallelism
- **Distributed memory**: Domain decomposition with MPI
- **GPU acceleration**: CUDA kernels for matrix assembly

### 5.3 Hybrid Methods

Combine with other techniques:
- **Physical optics**: High-frequency asymptotic for large structures
- **Finite elements**: Volume discretization for penetrable objects
- **Fast multipole**: $O(N \log N)$ complexity for large problems

---

## 6) Code Mapping

### 6.1 Primary Implementation Files

- **EFIE operator assembly**: `src/assembly/EFIE.jl`
  - `assemble_Z_efie`, singularity extraction, quadrature

- **Impedance loading**: `src/assembly/Impedance.jl` + `src/solver/Solve.jl`
  - `precompute_patch_mass`, `assemble_Z_impedance`, `assemble_full_Z`

- **Excitation assembly**: `src/assembly/Excitation.jl`
  - `make_plane_wave`, `assemble_excitation`, incident field integration

- **Linear algebra solvers**: `src/solver/Solve.jl`
  - `solve_forward`, `solve_system`
  - `select_preconditioner`, `prepare_conditioned_system`

- **Field postprocessing**: `src/postprocessing/NearField.jl`, `src/postprocessing/FarField.jl`
  - `compute_nearfield`, `compute_total_field`, `compute_farfield`

- **Diagnostics**: `src/postprocessing/Diagnostics.jl`
  - `condition_diagnostics`, `energy_ratio`, residual checks

### 6.2 Supporting Modules

- **Basis functions**: `src/basis/RWG.jl` (RWG construction)
- **Geometry utilities**: `src/geometry/Mesh.jl` (triangle operations)
- **Green kernels and quadrature**: `src/basis/Greens.jl`, `src/basis/Quadrature.jl`

### 6.3 Example Scripts

- **Forward solve / convergence**: `validation/paper/run_convergence_study.jl`
- **Beam objective workflow**: `examples/03_beamsteering_physical_unitcell.jl`
- **Near-/total-field validation**: `examples/21_near_total_field_rayleigh_sphere.jl`
- **Complex mesh forward solve**: `examples/06_aircraft_rcs.jl`

---

## 7) Exercises

### 7.1 Basic Level

1. **PEC plate analysis**:
   - Solve scattering from a $1\lambda \times 1\lambda$ plate at 3 GHz
   - Verify residual $< 10^{-8}$ and energy ratio $0.99 < \text{ratio} < 1.01$
   - Plot current magnitude distribution

2. **Impedance loading effects**:
   - Compare PEC vs. reactive impedance $Z_s = i100\,\Omega$
   - Analyze current reduction and phase shift
   - Verify impedance term correctness via matrix inspection

### 7.2 Intermediate Level

3. **Conditioning study**:
   - Solve plate problem with mesh refinement $N_x = 4, 8, 16$
   - Plot condition number vs. element size (log-log)
   - Enable preconditioning and quantify improvement

4. **Excitation variations**:
   - Solve for oblique incidence ($\theta_{\text{inc}} = 30^\circ$)
   - Compare TE vs. TM polarization
   - Verify reciprocity: $\mathbf{Z}$ symmetric within tolerance

### 7.3 Advanced Level

5. **Broadband analysis**:
   - Implement frequency sweep 1–10 GHz
   - Use frequency interpolation to reduce solve count
   - Compare interpolated vs. exact solution at mid-band

6. **Custom impedance distribution**:
   - Implement spatially varying impedance $Z_s(x,y)$
   - Verify gradient consistency with finite differences
   - Optimize distribution for beam steering

---

## 8) Chapter Checklist

Before proceeding to far-field computation or optimization, ensure you can:

- [ ] Assemble EFIE matrix for PEC and impedance-loaded surfaces
- [ ] Compute excitation vectors for arbitrary plane waves
- [ ] Solve linear systems with appropriate numerical methods
- [ ] Validate solutions through residuals and energy conservation
- [ ] Apply preconditioning for ill-conditioned problems
- [ ] Estimate memory requirements and computational complexity
- [ ] Troubleshoot common assembly and solution errors

---

## 9) Further Reading

- **Integral equation methods**: Chew et al., *Fast and Efficient Algorithms in Computational Electromagnetics* (2001)
- **RWG basis functions**: Rao et al., *Electromagnetic Scattering by Surfaces of Arbitrary Shape* (1982)
- **Numerical integration**: Graglia et al., *Singular Integrals in the Method of Moments* (1997)
- **Preconditioning techniques**: Andriulli et al., *Calderón Preconditioning for the EFIE* (2008)
- **Iterative methods**: Saad, *Iterative Methods for Sparse Linear Systems* (2003)
- **Broadband algorithms**: Ergin et al., *Fast Broadband Solution of Scattering Problems* (1999)
