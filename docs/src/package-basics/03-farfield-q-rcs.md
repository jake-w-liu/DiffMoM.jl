# Chapter 3: Field Postprocessing, Q Matrix, and RCS Computation

## Purpose

Transform surface current solutions into physically interpretable observables: scattered near fields, total electric fields, far-field patterns, directional power objectives via $\mathbf{Q}$ matrices, and radar cross section (RCS) quantities. This chapter establishes the mathematical relationships between MoM currents and field observables, provides efficient computational algorithms, and demonstrates practical workflows for scattering analysis and beam-steering objectives.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive the far-field radiation from RWG basis functions and surface currents.
2. Understand the mixed-potential scattered electric field used for near-field evaluation.
3. Interpret total electric field evaluation as `E_inc + E_sca` for supported excitation models.
4. Construct $\mathbf{Q}$ matrices for arbitrary directional and polarization objectives.
5. Compute bistatic and monostatic RCS with proper normalization and units.
6. Design spherical sampling grids for accurate power integration.
7. Implement beam-centric analysis for antenna and metasurface applications.
8. Validate field postprocessing against analytical benchmarks.
9. Optimize computational performance for large-scale radiation problems.

---

## 1) Mathematical Foundations

### 1.1 Far-Field Radiation from Surface Currents

The electric far-field $\mathbf{E}^\infty(\hat{\mathbf{r}})$ at direction $\hat{\mathbf{r}}$ (unit vector) produced by surface current $\mathbf{J}(\mathbf{r}')$ is:

```math
\mathbf{E}^\infty(\hat{\mathbf{r}}) = \frac{i k \eta_0}{4\pi} \int_\Gamma \left[ \hat{\mathbf{r}} \times \left( \hat{\mathbf{r}} \times \mathbf{J}(\mathbf{r}') \right) \right] e^{+i k \hat{\mathbf{r}} \cdot \mathbf{r}'} d\Gamma'
```

where $k = 2\pi/\lambda$ is the wavenumber and $\eta_0 = \sqrt{\mu_0/\epsilon_0} \approx 376.73\,\Omega$ is free-space impedance. This matches Equation (17) in `bare_jrnl.tex` (far‑field operator). Note that the factor $e^{ikr}/r$ describing spherical wave decay is omitted because $\mathbf{E}^\infty$ denotes the **far-field pattern** (amplitude and phase) independent of distance $r$. The complete far-field at large distance is $\mathbf{E}(\mathbf{r}) \approx \mathbf{E}^\infty(\hat{\mathbf{r}}) e^{ikr}/r$.

### 1.2 RWG Basis Expansion

Expanding the current in RWG basis functions $\mathbf{f}_n(\mathbf{r})$:

```math
\mathbf{J}(\mathbf{r}') = \sum_{n=1}^N I_n \mathbf{f}_n(\mathbf{r}')
```

yields a linear superposition:

```math
\mathbf{E}^\infty(\hat{\mathbf{r}}) = \sum_{n=1}^N I_n \mathbf{g}_n(\hat{\mathbf{r}})
```

where the **radiation vector** of basis $n$ is:

```math
\mathbf{g}_n(\hat{\mathbf{r}}) = \frac{i k \eta_0}{4\pi} \int_\Gamma \left[ \hat{\mathbf{r}} \times \left( \hat{\mathbf{r}} \times \mathbf{f}_n(\mathbf{r}') \right) \right] e^{+i k \hat{\mathbf{r}} \cdot \mathbf{r}'} d\Gamma'
```

### 1.3 Power and Directivity

The time-averaged Poynting vector in the far-field is:

```math
\mathbf{S}(\hat{\mathbf{r}}) = \frac{1}{2\eta_0} |\mathbf{E}^\infty(\hat{\mathbf{r}})|^2 \hat{\mathbf{r}}
```

Total radiated power:

```math
P_{\text{rad}} = \oint_{\mathbb{S}^2} \mathbf{S}(\hat{\mathbf{r}}) \cdot d\mathbf{A} = \frac{1}{2\eta_0} \int_{\mathbb{S}^2} |\mathbf{E}^\infty(\hat{\mathbf{r}})|^2 d\Omega
```

Directivity in direction $\hat{\mathbf{r}}$:

```math
D(\hat{\mathbf{r}}) = 4\pi \frac{|\mathbf{E}^\infty(\hat{\mathbf{r}})|^2}{\int_{\mathbb{S}^2} |\mathbf{E}^\infty(\hat{\mathbf{r}}')|^2 d\Omega'}
```

### 1.4 Quadratic Form for Directional Power

The power radiated into a region $\mathcal{D} \subset \mathbb{S}^2$ with polarization $\mathbf{p}(\hat{\mathbf{r}})$ is:

```math
P_{\mathcal{D}} = \frac{1}{2\eta_0} \int_{\mathcal{D}} |\mathbf{p}^\dagger(\hat{\mathbf{r}}) \mathbf{E}^\infty(\hat{\mathbf{r}})|^2 d\Omega
```

Substituting the current expansion:

```math
P_{\mathcal{D}} = \mathbf{I}^\dagger \mathbf{Q} \mathbf{I}
```

where $\mathbf{Q}$ is the **objective matrix** with elements:

```math
Q_{mn} = \frac{1}{2\eta_0} \int_{\mathcal{D}} \left[ \mathbf{p}^\dagger(\hat{\mathbf{r}}) \mathbf{g}_m(\hat{\mathbf{r}}) \right]^* \left[ \mathbf{p}^\dagger(\hat{\mathbf{r}}) \mathbf{g}_n(\hat{\mathbf{r}}) \right] d\Omega
```

**Paper reference:** Equation (18) in `bare_jrnl.tex` defines the Q‑matrix entries.

### 1.5 Ratio Objective for Beam Steering

For beam-steering applications, maximizing absolute radiated power in a target region can lead to undesirable beam broadening. Instead, we maximize the **directivity fraction**—the ratio of power in the target cone to total radiated power in the same polarization:

```math
J(\boldsymbol{\theta}) = \frac{\mathbf{I}^\dagger \mathbf{Q}_{\text{target}} \mathbf{I}}{\mathbf{I}^\dagger \mathbf{Q}_{\text{total}} \mathbf{I}}
```

where $\mathbf{Q}_{\text{total}}$ integrates over the entire sphere $\mathbb{S}^2$ using the same polarization projector as $\mathbf{Q}_{\text{target}}$. This ratio objective is invariant to global scaling of currents and naturally penalizes radiation outside the target region.

**Gradient via quotient rule:** Differentiating $J = f/g$ where $f = \mathbf{I}^\dagger \mathbf{Q}_{\text{target}} \mathbf{I}$ and $g = \mathbf{I}^\dagger \mathbf{Q}_{\text{total}} \mathbf{I}$ yields:

```math
\frac{\partial J}{\partial \theta_p} = \frac{1}{g^2} \left( g \frac{\partial f}{\partial \theta_p} - f \frac{\partial g}{\partial \theta_p} \right)
```

Each term $\partial f/\partial \theta_p$ and $\partial g/\partial \theta_p$ requires its own adjoint solve (see Part III for the full adjoint formulation). This two‑solve approach avoids numerical cancellation errors that arise when using the Dinkelbach transformation $\mathbf{Q}_{\text{eff}} = \mathbf{Q}_{\text{target}} - J \mathbf{Q}_{\text{total}}$, which satisfies $\mathbf{I}^\dagger \mathbf{Q}_{\text{eff}} \mathbf{I} \approx 0$ at convergence and leads to loss of precision in gradient evaluation.

**Paper reference:** Equation (19) in `bare_jrnl.tex` defines the ratio objective, and Section III‑F details the two‑adjoint quotient‑rule gradient used for beam‑steering optimization.

### 1.6 Radar Cross Section (RCS)

The bistatic RCS for incident plane wave $\mathbf{E}^{\text{inc}}$ is:

```math
\sigma(\hat{\mathbf{r}}) = 4\pi \lim_{r\to\infty} r^2 \frac{|\mathbf{E}^\infty(\hat{\mathbf{r}})|^2}{|\mathbf{E}^{\text{inc}}|^2}
```

For backscatter ($\hat{\mathbf{r}} = -\hat{\mathbf{k}}_{\text{inc}}$), this becomes monostatic RCS.

### 1.7 Scattered Near-Field from Surface Currents

Away from the surface, the solved RWG current coefficients also define the
scattered electric field at arbitrary observation points:

```math
\mathbf E^{\mathrm{sca}}(\mathbf r)
=
-i k \eta_0 \int_\Gamma \mathbf J(\mathbf r') G(\mathbf r,\mathbf r')\, dS'
-i \frac{\eta_0}{k} \int_\Gamma \big(\nabla'\cdot\mathbf J(\mathbf r')\big)\, \nabla G(\mathbf r,\mathbf r')\, dS'.
```

This mixed-potential representation is the same physical field model used by
the EFIE. In `DiffMoM.jl`, `compute_nearfield` evaluates this
expression numerically for any off-surface observation point.

### 1.8 Total Electric Field

For supported excitation models, the total electric field is defined pointwise by

```math
\mathbf E^{\mathrm{tot}}(\mathbf r) = \mathbf E^{\mathrm{inc}}(\mathbf r) + \mathbf E^{\mathrm{sca}}(\mathbf r).
```

The package computes `E_inc` from the supplied `AbstractExcitation` object.
This is available for pointwise source models such as plane waves, dipoles,
loops, pattern feeds, imported electric fields, and supported
`MultiExcitation` combinations. Edge-localized feed surrogates such as ports
and delta-gap excitations are intentionally excluded from total-field
observation.

### 1.9 Adjoint Gradient for Q Matrix

The gradient of a quadratic objective $\Phi = \mathbf{I}^\dagger \mathbf{Q} \mathbf{I}$ with respect to design parameters $\boldsymbol{\theta}$ follows from the adjoint method derived in Part III. For a single quadratic objective, the adjoint variable $\boldsymbol{\lambda}$ solves

```math
\mathbf{Z}^\dagger \boldsymbol{\lambda} = \mathbf{Q} \mathbf{I}
```

and the gradient components are

```math
\frac{\partial \Phi}{\partial \theta_p} = -2 \Re\!\left\{ \boldsymbol{\lambda}^\dagger \left( \frac{\partial \mathbf{Z}}{\partial \theta_p} \right) \mathbf{I} \right\}.
```

For ratio objectives $J = f/g$, the quotient rule (Section 1.5) requires two adjoint solves:

```math
\mathbf{Z}^\dagger \boldsymbol{\lambda}_f = \mathbf{Q}_{\text{target}} \mathbf{I}, \qquad
\mathbf{Z}^\dagger \boldsymbol{\lambda}_g = \mathbf{Q}_{\text{total}} \mathbf{I}.
```

The gradient of the ratio is then assembled from the individual gradients $\partial f/\partial \theta_p$ and $\partial g/\partial \theta_p$ via the quotient formula. This two‑solve approach preserves numerical accuracy and is used in the beam‑steering example of Section 3.2.

**Paper reference:** Equation (25) in `bare_jrnl.tex` gives the adjoint gradient for a quadratic objective, and Section III‑F details the two‑adjoint quotient‑rule gradient used for beam‑steering optimization.

---

## 2) Implementation in `DiffMoM.jl`

### 2.1 Spherical Grid Construction

Discrete angular sampling on the unit sphere:

```julia
grid = make_sph_grid(Nθ, Nφ)
```

**Grid properties:**
- **Midpoint sampling**: $\theta$ and $\phi$ at cell midpoints (half-step offset from boundaries)
- **Quadrature weights**: $w_q = \sin\theta_q \Delta\theta \Delta\phi$ for accurate integration
- **Total solid angle**: $\sum_{q=1}^{N_\Omega} w_q \approx 4\pi$
- **Vectorization**: Column-major storage for efficient computation

**Recommended resolutions:**
- **Quick visualization**: $N_\theta = 30$, $N_\phi = 60$
- **Accurate integration**: $N_\theta = 60$, $N_\phi = 120$
- **High-fidelity patterns**: $N_\theta = 180$, $N_\phi = 360$

### 2.2 Radiation Vector Precomputation

Precompute radiation vectors for all basis functions and directions:

```julia
G_mat = radiation_vectors(mesh, rwg, grid, k;
    quad_order=3,
    eta0=η0)
```

**Algorithm (see `src/postprocessing/FarField.jl`):** For each RWG basis function $\mathbf{f}_n$ with support triangles $T_n^+$ and $T_n^-$, compute

```math
\mathbf{g}_n(\hat{\mathbf{r}}_q) = \frac{i k \eta_0}{4\pi} \sum_{t \in \{T_n^+, T_n^-\}} \int_{T_t} \left[ \hat{\mathbf{r}}_q \times \left( \hat{\mathbf{r}}_q \times \mathbf{f}_n(\mathbf{r}') \right) \right] e^{+i k \hat{\mathbf{r}}_q \cdot \mathbf{r}'} d\Gamma'
```

using the same triangular quadrature rule (order `quad_order`) as EFIE assembly. The integral is evaluated at quadrature points $\mathbf{r}_p'$ with weights $w_p$, and the phase factor $e^{+i k \hat{\mathbf{r}}_q \cdot \mathbf{r}_p'}$ is applied per direction. The outer product $\hat{\mathbf{r}} \times (\hat{\mathbf{r}} \times \mathbf{f}_n)$ simplifies to $(\hat{\mathbf{r}} \cdot \mathbf{f}_n) \hat{\mathbf{r}} - \mathbf{f}_n$.

**Implementation details:** The function `radiation_vectors` loops over basis functions $n$, over their two support triangles, over quadrature points within each triangle, and over all directions $\hat{\mathbf{r}}_q$. For performance, the direction loop is innermost, allowing vectorization across directions. The result is stored as a dense matrix `G_mat` of size $(3 N_\Omega) \times N$, where each column $n$ contains the three Cartesian components of $\mathbf{g}_n$ for all directions concatenated.

**Memory considerations:** 
- Storage: $3 \times N_\Omega \times N$ complex numbers
- Example: $N=10,000$, $N_\Omega=21,600$ (180×120) → ~5 GB (double precision)
- Use `Float32` or sparse representation for large problems
- On‑the‑fly computation trades memory for recomputation time

### 2.3 Far-Field Evaluation

Compute far-field at all directions:

```julia
E_ff = compute_farfield(G_mat, I, NΩ)
```

**Implementation:** Matrix-vector product $\mathbf{E}^\infty = \mathbf{G} \mathbf{I}$

**Efficient recomputation:** For multiple current vectors $\mathbf{I}^{(k)}$, reuse $\mathbf{G}$:

```julia
# Precompute G once
G_mat = radiation_vectors(...)

# Evaluate for multiple current solutions
for (idx, I_k) in enumerate(currents)
    E_k = compute_farfield(G_mat, I_k, NΩ)
end
```

### 2.4 Q Matrix Construction

Build objective matrix for directional power:

```julia
# Polarization vectors (e.g., linear x-polarization)
pol = pol_linear_x(grid)  # Size: 3 × NΩ

# Angular mask (e.g., cone around broadside)
mask = cap_mask(grid; theta_max=30π/180)  # Boolean vector

# Construct Q matrix
Q = build_Q(G_mat, grid, pol; mask=mask)
```

**Mathematical implementation:**

```math
\mathbf{Q} = \sum_{q=1}^{N_\Omega} w_q \cdot m_q \cdot \mathbf{v}_q \mathbf{v}_q^\dagger
```

where:
- $w_q$: spherical quadrature weight from `grid.w`
- $m_q$: mask value (0 or 1) selecting directions in the target region
- $\mathbf{v}_q = \mathbf{G}[:,q]^\dagger \mathbf{p}_q$ (projected radiation vectors), with $\mathbf{G}[:,q]$ the $3\times N$ slice of radiation vectors for direction $q$.

**Implementation (see `src/optimization/QMatrix.jl`):** The function `build_Q` computes the scalar projections $y_{q,n} = \mathbf{p}_q^\dagger \mathbf{g}_n(\hat{\mathbf{r}}_q)$ for each direction $q$ and basis $n$, then accumulates $Q_{mn} = \sum_q w_q m_q \, y_{q,m}^* y_{q,n}$. The double sum over $m,n$ is explicit but optimized with nested loops. For large problems, the alternative function `apply_Q` computes $\mathbf{Q}\mathbf{I}$ without forming $\mathbf{Q}$ explicitly, using the factorization $\mathbf{Q} = \mathbf{Y}^\dagger \mathbf{W} \mathbf{Y}$ where $\mathbf{Y}$ is the $N_\Omega \times N$ matrix of projected radiation vectors and $\mathbf{W}$ is a diagonal matrix of weighted mask values.

**Properties:**
- **Hermitian**: $\mathbf{Q} = \mathbf{Q}^\dagger$
- **Positive semidefinite**: $\mathbf{I}^\dagger \mathbf{Q} \mathbf{I} \geq 0$
- **Sparse structure**: Dense in general, but can be low-rank approximated

**Objective evaluation:**

```julia
J = compute_objective(I, Q)  # = real(dot(I, Q * I))
```

**Ratio objective:** For beam‑steering applications, the directivity fraction $J = (\mathbf{I}^\dagger \mathbf{Q}_{\text{target}} \mathbf{I}) / (\mathbf{I}^\dagger \mathbf{Q}_{\text{total}} \mathbf{I})$ is evaluated by computing both quadratic forms separately. The gradient is assembled via the quotient rule using two adjoint solves (see Section 1.5). The beam‑steering example `examples/03_beamsteering_physical_unitcell.jl` demonstrates this workflow.

### 2.5 RCS Computation

#### Bistatic RCS
```julia
σ_bistatic = bistatic_rcs(E_ff; E0=1.0)
```

Implementation:
```math
\sigma_q = 4\pi \frac{|\mathbf{E}^\infty(\hat{\mathbf{r}}_q)|^2}{|E_0|^2}
```

#### Backscatter RCS
```julia
bs_result = backscatter_rcs(E_ff, grid, k_inc_hat; E0=1.0)
# Returns: sigma, index, theta, phi, angular_error_deg
```

Finds direction closest to $-\hat{\mathbf{k}}_{\text{inc}}$ and returns RCS value.

#### dB Conversion
```julia
σ_dB = 10 .* log10.(max.(σ_bistatic, 1e-30))  # dBsm
```

### 2.6 Polarization Handling

The package supports arbitrary polarization vectors:

```julia
# Built-in linear-θ polarization (x-pol / TM for φ=0 plane)
pol_x = pol_linear_x(grid)

# Built-in linear-φ polarization (y-pol / TE for φ=0 plane)
pol_y = pol_linear_y(grid)

# User-defined polarization matrix (3 × NΩ), column q = p(r̂_q)
pol_custom = zeros(ComplexF64, 3, length(grid.w))
for q in eachindex(grid.w)
    θ = grid.theta[q]
    φ = grid.phi[q]
    theta_hat = Vec3(cos(θ)*cos(φ), cos(θ)*sin(φ), -sin(θ))
    pol_custom[:, q] = theta_hat
end
```

### 2.7 Angular Masks

Select specific angular regions:

```julia
# Cone around broadside
mask_cone = cap_mask(grid; theta_max=30π/180)

# Custom mask function
mask_custom = [abs(φ) < π/4 && θ > π/6 for (θ, φ) in zip(grid.theta, grid.phi)]

# Combine masks
mask_combined = mask_cone .& mask_custom
```

### 2.8 Near-Field and Total-Field Evaluation

Postprocess the solved currents at arbitrary off-surface points:

```julia
pw = make_plane_wave(Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
obs = [Vec3(0.0, 0.0, 0.15), Vec3(0.02, 0.0, 0.18)]

E_sca = compute_nearfield(mesh, rwg, I, obs, k; quad_order=3, eta0=η0)
E_tot = compute_total_field(mesh, rwg, I, pw, obs, k; quad_order=3, eta0=η0)
```

`compute_nearfield` evaluates only the scattered field. `compute_total_field`
adds the pointwise incident field from the supplied excitation object, so the
excitation must be one of the supported observation-space source models.

Implementation lives in `src/postprocessing/NearField.jl` and complements the
far-field operator path in `src/postprocessing/FarField.jl`.

---

## 3) Practical Workflow Examples

### 3.1 Complete Field Postprocessing Analysis

```julia
using DiffMoM

# 1. Solve forward problem (as in Chapter 2)
mesh = make_rect_plate(0.1, 0.1, 8, 8)
rwg = build_rwg(mesh)
f = 3e9
k = 2π * f / 299792458.0
Z = assemble_Z_efie(mesh, rwg, k)
pw = make_plane_wave(Vec3(0,0,-k), 1.0, Vec3(1,0,0))
v = assemble_excitation(mesh, rwg, pw)
I = solve_forward(Z, v)

# 1b. Sample local fields
obs = [Vec3(0.0, 0.0, 0.15), Vec3(0.02, 0.0, 0.18)]
E_sca = compute_nearfield(mesh, rwg, I, obs, k)
E_tot = compute_total_field(mesh, rwg, I, pw, obs, k)
println("First scattered-field sample: ", E_sca[:, 1])
println("First total-field sample: ", E_tot[:, 1])

# 2. Create spherical grid
grid = make_sph_grid(90, 180)  # 90×180 = 16,200 directions
println("Grid size: $(length(grid.w)) samples")
println("Total solid angle: $(sum(grid.w)) (should be ≈ $(4π))")

# 3. Precompute radiation vectors
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3)

# 4. Compute far-field
E_ff = compute_farfield(G_mat, I, length(grid.w))

# 5. Compute RCS
σ = bistatic_rcs(E_ff; E0=1.0)
σ_dB = 10 .* log10.(max.(σ, 1e-30))

# 6. Extract main beam statistics
max_idx = argmax(σ_dB)
θ_max = grid.theta[max_idx] * 180/π
φ_max = grid.phi[max_idx] * 180/π
println("Main beam direction: θ=$(θ_max)°, φ=$(φ_max)°")
println("Peak RCS: $(σ_dB[max_idx]) dBsm")
```

### 3.2 Directional Objective for Beam Steering

```julia
# Target direction
θ_target = 30π/180
φ_target = 0.0

# Create mask around target (10° cone)
function target_mask(grid, θ0, φ0, cone_angle=10π/180)
    mask = zeros(Bool, length(grid.w))
    for q in 1:length(grid.w)
        θ = grid.theta[q]
        φ = grid.phi[q]
        # Angular distance on sphere
        dist = acos(sin(θ)*sin(θ0)*cos(φ-φ0) + cos(θ)*cos(θ0))
        mask[q] = dist < cone_angle
    end
    return mask
end

mask_target = target_mask(grid, θ_target, φ_target, 10π/180)

# Build Q matrix for target region
pol = pol_linear_x(grid)
Q_target = build_Q(G_mat, grid, pol; mask=mask_target)

# Compute power in target region
P_target = compute_objective(I, Q_target)

# For comparison, compute total radiated power
mask_total = trues(length(grid.w))
Q_total = build_Q(G_mat, grid, pol; mask=mask_total)
P_total = compute_objective(I, Q_total)

# Directivity fraction
directivity_fraction = P_target / P_total
println("Power in target cone: $(directivity_fraction*100)% of total")
```

### 3.3 RCS Pattern Analysis

```julia
# Extract φ=0° cut (E-plane)
φ_cut = 0.0
tolerance = 1e-6
cut_indices = [q for q in 1:length(grid.w) 
               if abs(grid.phi[q] - φ_cut) < tolerance]
sort_idx = sortperm(grid.theta[cut_indices])
θ_cut = grid.theta[cut_indices][sort_idx]
σ_cut = σ_dB[cut_indices][sort_idx]

# Find main lobe and sidelobes
function analyze_pattern(θ_deg, σ_dB)
    # Find global maximum
    main_idx = argmax(σ_dB)
    θ_main = θ_deg[main_idx]
    σ_main = σ_dB[main_idx]
    
    # Find sidelobes (excluding ±10° around main lobe)
    sidelobe_mask = abs.(θ_deg .- θ_main) .> 10.0
    if any(sidelobe_mask)
        sidelobe_idx = argmax(σ_dB[sidelobe_mask])
        θ_sl = θ_deg[sidelobe_mask][sidelobe_idx]
        σ_sl = σ_dB[sidelobe_mask][sidelobe_idx]
        sll = σ_main - σ_sl  # sidelobe level
    else
        θ_sl = NaN
        σ_sl = NaN
        sll = NaN
    end
    
    return (θ_main=θ_main, σ_main=σ_main, 
            θ_sl=θ_sl, σ_sl=σ_sl, sll=sll)
end

stats = analyze_pattern(θ_cut * 180/π, σ_cut)
println("Main lobe: $(stats.θ_main)°, $(stats.σ_main) dBsm")
println("Strongest sidelobe: $(stats.θ_sl)°, $(stats.σ_sl) dBsm")
println("Sidelobe level: $(stats.sll) dB")
```

### 3.4 Validation Against Analytical Solution

**Far-field / RCS reference:** The full sphere‑vs‑Mie benchmark is presented in Part V, Chapter 4 (Sphere‑vs‑Mie Benchmark). That chapter provides detailed error metrics, convergence studies, and validation against the analytical Mie series.

**Near-field / total-field reference:** The [Rayleigh small-sphere benchmark](../validation/06-near-total-field-rayleigh-sphere.md) validates `compute_nearfield` and `compute_total_field` against an analytical dipole-limit reference. Use that benchmark, not physical optics, when validating local electric fields in the electrically small regime.

```julia
# Compare with Mie theory for PEC sphere (functions available via the package)
using DiffMoM

# Sphere parameters
a = 0.05  # 5 cm radius
k = 2π * 3e9 / 299792458.0

# MoM solution (from sphere benchmark)
I_mom = ...  # obtained from sphere solve

# Compute far-field
grid = make_sph_grid(181, 72)  # Match Mie benchmark resolution
G_mat = radiation_vectors(mesh, rwg, grid, k)
E_mom = compute_farfield(G_mat, I_mom, length(grid.w))
σ_mom = bistatic_rcs(E_mom; E0=1.0)

# Mie theory reference
σ_mie = zeros(length(grid.w))
for q in 1:length(grid.w)
    rhat = Vec3(grid.rhat[:, q])
    σ_mie[q] = mie_bistatic_rcs_pec(k, a, Vec3(0,0,-1), Vec3(1,0,0), rhat)
end

# Compute error
error_dB = 10*log10.(max.(σ_mom, 1e-30)) - 10*log10.(max.(σ_mie, 1e-30))
mae = mean(abs.(error_dB))
println("Mean absolute error: $mae dB")
```

### 3.5 Beam-Steering Optimization Workflow

The complete beam‑steering inverse‑design pipeline is implemented in `examples/03_beamsteering_physical_unitcell.jl`. This example optimizes a reactive impedance sheet on a $4\lambda\times4\lambda$ plate to steer a normally‑incident plane wave to $\theta_s = 30^\circ$ off broadside.

**Key steps from the script:**

1. **Setup:** Create plate mesh, assemble EFIE matrix, define impedance patches.
2. **Far‑field grid:** Build spherical grid with $1^\circ$ $\theta$‑resolution.
3. **Target mask:** Select directions within a $5^\circ$ cone around $\theta_s$.
4. **Q matrices:** Construct $\mathbf{Q}_{\text{target}}$ and $\mathbf{Q}_{\text{total}}$ for x‑polarization.
5. **Reference PEC solution:** Compute directivity fraction $J_{\text{PEC}}$.
6. **Optimization:** Maximize $J$ using projected L‑BFGS with reactive impedance constraints $|\theta_p| \le 500\,\Omega$.
7. **Post‑processing:** Evaluate optimized far‑field pattern, compute gradient verification.

**Results from the paper (`bare_jrnl.tex`):** The optimization achieves a $28^\circ$ beam shift and a $23$ dB gain in the target direction relative to the PEC baseline. The directivity fraction $J$ improves from $0.8\%$ (PEC) to $18.5\%$ (optimized), demonstrating effective beam steering.

**Gradient verification:** The script includes finite‑difference checks of the adjoint gradient at the optimum, confirming gradient accuracy to within $10^{-6}$ relative error.

**Running the example:**
```bash
julia --project=. examples/03_beamsteering_physical_unitcell.jl
```
Output CSV files are saved to `data/` for further analysis.

### 3.6 Airplane RCS Demonstration

The script `examples/06_aircraft_rcs.jl` demonstrates RCS computation for a complex CAD geometry (an airplane model). It includes automatic mesh repair, optional coarsening, and far‑field analysis.

**Key features:**
- **Mesh repair:** Automatically fixes orientation, removes degenerate triangles, and ensures manifold edges.
- **Coarsening:** Reduces mesh density to a practical RWG count while preserving shape.
- **Far‑field and RCS:** Computes bistatic and monostatic RCS on a spherical grid.
- **Visualization hooks:** Can be extended with `save_mesh_preview` in custom scripts.

**Typical usage:**
```bash
julia --project=. examples/06_aircraft_rcs.jl
```
This demo script currently takes no CLI arguments; edit the script variables if
you need custom geometry or frequency.

**Outputs:** The current script prints solver and RCS diagnostics to stdout.
You can add CSV/OBJ/plot exports in a short post-processing block.

This example illustrates the end‑to‑end pipeline for realistic platform scattering analysis.

---

## 4) Performance Optimization

### 4.1 Memory-Efficient Strategies

For large problems ($N > 20,000$, $N_\Omega > 10,000$):

```julia
# 1. Use single precision
G_mat_single = Float32.(G_mat)

# 2. Compute on-the-fly (trade memory for computation)
function farfield_on_the_fly(mesh, rwg, grid, k, I)
    E = zeros(ComplexF64, 3, length(grid.w))
    for q in 1:length(grid.w)
        rhat = Vec3(grid.rhat[:, q])
        # Compute radiation vectors for this direction only
        g_vec = compute_radiation_vector_direction(mesh, rwg, k, rhat)
        E[:, q] = g_vec * I
    end
    return E
end

# 3. Block processing
function farfield_block(G_mat, I, block_size=1000)
    NΩ = size(G_mat, 2)
    E = zeros(ComplexF64, 3, NΩ)
    for start_idx in 1:block_size:NΩ
        end_idx = min(start_idx + block_size - 1, NΩ)
        G_block = @view G_mat[:, start_idx:end_idx]
        E_block = G_block * I
        E[:, start_idx:end_idx] .= E_block
    end
    return E
end
```

### 4.2 Accelerated Computation

```julia
# 1. Multi-threading
using Base.Threads
function farfield_threaded(G_mat, I)
    NΩ = size(G_mat, 2)
    E = zeros(ComplexF64, 3, NΩ)
    @threads for q in 1:NΩ
        E[:, q] = @view(G_mat[:, q]) * I
    end
    return E
end

# 2. GPU acceleration (requires CUDA.jl)
using CUDA
function farfield_gpu(G_mat, I)
    G_gpu = CuArray(G_mat)
    I_gpu = CuArray(I)
    E_gpu = G_gpu * I_gpu
    return Array(E_gpu)
end
```

### 4.3 Approximation Techniques

```julia
# 1. Low-rank approximation of G matrix
function lowrank_approximate(G_mat, rank)
    U, Σ, V = svd(G_mat, rank)
    return U, Diagonal(Σ), V
end

# 2. Adaptive sampling (more samples in high-variation regions)
function adaptive_grid(mesh, k, max_samples=10000)
    # Sample density proportional to expected pattern complexity
    # Higher density near specular directions, edges, etc.
    # Implementation depends on geometry
end
```

---

## 5) Troubleshooting Common Issues

### 5.1 Diagnostic Decision Tree

1. **Far-field pattern shows unexpected symmetry breaking**
   - Check mesh symmetry and excitation alignment
   - Verify polarization vector definition
   - Inspect numerical noise (increase quadrature order)

2. **Q matrix objective doesn't match direct integration**
   - Verify spherical grid weights: `sum(grid.w) ≈ 4π`
   - Check mask alignment with grid samples
   - Confirm polarization projection consistency

3. **RCS values unrealistic (extremely large or small)**
   - Verify incident field magnitude `E0`
   - Check units (meters for geometry, Hz for frequency)
   - Confirm far-field scaling factor $\frac{ik\eta_0}{4\pi}$

4. **Memory exhaustion during G matrix computation**
   - Reduce angular resolution
   - Use on-the-fly computation
   - Employ single precision or compression

5. **Poor convergence of optimization with Q objectives**
   - Verify Q matrix is Hermitian (numerical symmetry)
   - Check condition number of Q (may need regularization)
   - Ensure mask covers sufficient solid angle for stable gradient

### 5.2 Validation Procedures

Establish validation protocol for far-field computations:

1. **Power conservation**: $\mathbf{I}^\dagger \mathbf{Q}_{\text{total}} \mathbf{I} \approx P_{\text{rad}}$ from Poynting integration
2. **Reciprocity**: Far-field pattern should match for swapped source/observation
3. **Analytical benchmarks**: Sphere vs. Mie for far field / RCS, Rayleigh small-sphere for near / total fields, plate vs. physical optics for high-frequency approximation studies
4. **Grid convergence**: Monitor objective value with increasing $N_\theta$, $N_\phi$
5. **Polarization purity**: Verify orthogonal polarizations yield zero cross-power

### 5.3 Accuracy vs. Performance Trade-offs

- **Quick visualization**: $N_\theta=30$, $N_\phi=60$, single precision
- **Accurate integration**: $N_\theta=90$, $N_\phi=180$, double precision  
- **High-fidelity optimization**: $N_\theta=180$, $N_\phi=360$, with gradient validation
- **Large-scale problems**: Adaptive sampling, on-the-fly computation, GPU acceleration

---

## 6) Advanced Topics

### 6.1 Near-Field and Total-Field Evaluation

For local field probes, use the dedicated postprocessing routines:

```julia
pw = make_plane_wave(Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
obs = [Vec3(0.0, 0.0, 0.15), Vec3(0.02, 0.0, 0.18)]

E_sca = compute_nearfield(mesh, rwg, I, obs, k; quad_order=3, eta0=η0)
E_tot = compute_total_field(mesh, rwg, I, pw, obs, k; quad_order=3, eta0=η0)
```

Use `compute_nearfield` when you want the scattered field alone, such as for
field cancellation or equivalence-principle checks. Use `compute_total_field`
when you want the observable electric field in space:
`E_total = E_inc + E_sca`.

Current limitations:

- Observation points must be off the surface.
- Near-singular quadrature is not implemented yet, so very near-surface probes
  may require higher `quad_order`.
- `compute_total_field` supports only source models with a well-defined
  pointwise `E_inc(r)`.

### 6.2 Bistatic RCS for Multiple Incidences

```julia
function bistatic_rcs_matrix(mesh, rwg, k, inc_directions, obs_grid)
    # Compute RCS matrix for multiple incidence directions
    # Returns matrix σ[i,j] for incidence i, observation j
end
```

### 6.3 Partial Coherence and Stochastic Excitation

```julia
function partially_coherent_power(Q, covariance_matrix)
    # Compute expected power for stochastic currents
    # E[I† Q I] = tr(Q * covariance_matrix)
    return tr(Q * covariance_matrix)
end
```

### 6.4 Harmonic Expansion

Expand far-field in spherical harmonics:

```julia
using SphericalHarmonics

function spherical_harmonic_expansion(E_ff, grid, L_max)
    # Project far-field onto spherical harmonic basis
    # Useful for pattern synthesis and compression
end
```

---

## 7) Code Mapping

### 7.1 Primary Implementation Files

- **Far-field grid and radiation vectors**: `src/postprocessing/FarField.jl`
  - `make_sph_grid`, `radiation_vectors`, `compute_farfield`

- **Near-field and total-field evaluation**: `src/postprocessing/NearField.jl`
  - `compute_nearfield`, `compute_total_field`

- **Q matrix construction**: `src/optimization/QMatrix.jl`
  - `build_Q`, `apply_Q`, `pol_linear_x`, `pol_linear_y`, `cap_mask`, `direction_mask`

- **RCS diagnostics**: `src/postprocessing/Diagnostics.jl`
  - `bistatic_rcs`, `backscatter_rcs`, `radiated_power`

- **Mie theory reference**: `src/postprocessing/Mie.jl`
  - `mie_bistatic_rcs_pec`, analytical validation

- **Incident-field models for total-field evaluation**: `src/assembly/Excitation.jl`
  - `make_plane_wave`, `assemble_excitation`, supported excitation dispatch

### 7.2 Example Scripts

- **Sphere benchmark**: `examples/04_pec_sphere_mie.jl`
- **Beam steering optimization**: `examples/03_beamsteering_physical_unitcell.jl`
- **Near-/total-field Rayleigh benchmark**: `examples/21_near_total_field_rayleigh_sphere.jl`
- **Platform RCS**: `examples/06_aircraft_rcs.jl`

### 7.3 Supporting Utilities

- **Geometry operations**: `src/geometry/Mesh.jl` (triangle geometry helpers)
- **Visualization**: `src/postprocessing/Visualization.jl` (mesh plotting utilities)
- **Performance tools**: `src/geometry/Mesh.jl` (`estimate_dense_matrix_gib`)

---

## 8) Exercises

### 8.1 Basic Level

1. **Far-field pattern visualization**:
   - Compute and plot far-field pattern for PEC plate
   - Identify main beam direction and 3-dB beamwidth
   - Verify pattern symmetry for symmetric geometry/excitation

2. **Q matrix validation**:
   - Build Q matrix for broadside cone ($\theta < 30^\circ$)
   - Compare `I† Q I` with direct angular integration
   - Verify Hermitian and PSD properties numerically
3. **Incident-field recovery**:
   - Evaluate `E_sca` and `E_tot` at a few off-surface points
   - Verify numerically that `E_tot - E_sca` matches the incident field
   - Repeat with higher `quad_order` for points closer to the surface

### 8.2 Intermediate Level

4. **RCS accuracy study**:
   - Compute sphere RCS with increasing angular resolution
   - Plot error vs. grid size (convergence analysis)
   - Determine minimum grid for < 0.5 dB error

5. **Polarization analysis**:
   - Compute co-polar and cross-polar patterns
   - Verify polarization purity for linear/circular cases
   - Analyze polarization mismatch loss

### 8.3 Advanced Level

6. **Beam synthesis**:
   - Design Q matrix for shaped beam (cosecant, flat-top)
   - Optimize currents to match target pattern
   - Evaluate realizability with passive impedance sheets

7. **Computational optimization**:
   - Implement block-based far-field computation
   - Compare performance (memory, time) for different strategies
   - Propose hybrid approach for large-scale problems

---

## 9) Chapter Checklist

Before using field postprocessing quantities for design optimization, ensure you can:

- [ ] Compute accurate far-field patterns from surface currents
- [ ] Compute scattered near-field and total-field samples away from the surface
- [ ] Construct Q matrices for arbitrary directional objectives
- [ ] Calculate RCS with proper normalization and units
- [ ] Validate results against analytical benchmarks
- [ ] Optimize computational performance for your problem size
- [ ] Troubleshoot common accuracy and performance issues
- [ ] Interpret beam-centric metrics for antenna applications

---

## 10) Further Reading

- **Paper foundations**: The mathematical derivations of far‑field operators, Q‑matrix construction, and ratio objectives are detailed in `bare_jrnl.tex`, Sections III‑E and III‑F.
- **Antenna theory**: Balanis, *Antenna Theory: Analysis and Design* (2016)
- **RCS prediction**: Knott et al., *Radar Cross Section* (1993)
- **Spherical wave expansions**: Jackson, *Classical Electrodynamics* (1999)
- **Numerical integration on sphere**: Atkinson, *Numerical Integration on the Sphere* (1982)
- **Fast far-field algorithms**: Chew et al., *Fast and Efficient Algorithms in Computational Electromagnetics* (2001)
- **Polarimetry**: Mott, *Remote Sensing with Polarimetric Radar* (2007)
- **Beam shaping**: Haupt, *Antenna Arrays: A Computational Approach* (2010)
