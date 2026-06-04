# Multi-Level Fast Multipole Algorithm (MLFMA)

## Purpose

For very large problems ($N > 50{,}000$ unknowns), even H-matrix compression with ACA becomes limited by the $O(N \log^2 N)$ near-field storage cost. The Multi-Level Fast Multipole Algorithm (MLFMA) reduces both storage and matrix-vector product cost to $O(N \log N)$ by hierarchically decomposing electromagnetic interactions into near-field (directly computed) and far-field (computed via radiation patterns and translation operators) contributions.

This chapter explains the MLFMA implementation in `DiffMoM.jl`: octree construction, radiation pattern computation, multi-level aggregation and disaggregation with per-$m$ spectral filters, translation operators, near-field preconditioning, and practical usage guidelines including accuracy considerations.

---

## Learning Goals

After this chapter, you should be able to:

1. Explain why MLFMA achieves $O(N \log N)$ complexity for EFIE matrix-vector products.
2. Describe the octree spatial decomposition and its role in the MLFMA hierarchy.
3. Understand radiation pattern computation and the 4-component vector formulation.
4. Explain aggregation (upward pass), translation (interaction), and disaggregation (downward pass).
5. Describe the per-$m$ spectral filter implementation and why separated filters fail for $m \neq 0$ modes.
6. Use `build_mlfma_operator` and `solve_scattering` with MLFMA in practice.
7. Choose `leaf_lambda` with an accuracy-cost sweep (default in code: `leaf_lambda=0.25`).
8. Configure MLFMA preconditioning paths used by the workflow and optional reordered variants.

---

## 1. The Scalability Problem

### 1.1 Comparison of Methods

| Method | Storage | Matvec | Assembly | Best for |
|--------|---------|--------|----------|----------|
| Dense | $O(N^2)$ | $O(N^2)$ | $O(N^2)$ | $N < 2{,}000$ |
| ACA H-matrix | $O(N \log^2 N)$ | $O(N \log^2 N)$ | $O(N \log^2 N)$ | $2{,}000 < N < 50{,}000$ |
| **MLFMA** | **$O(N \log N)$** | **$O(N \log N)$** | **$O(N \log N)$** | **$N > 50{,}000$** |

In practice, the exact memory and runtime depend strongly on geometry, frequency, and parameter choices (`leaf_lambda`, `precision`, quadrature order, and preconditioner settings). Treat performance numbers as problem-specific and benchmark on your target case.

### 1.2 The MLFMA Idea

The key insight is that far-field electromagnetic interactions can be computed via:
1. **Aggregation**: Compute radiation patterns for groups of sources
2. **Translation**: Shift these patterns between distant groups (cheap in the far field)
3. **Disaggregation**: Convert received patterns to fields at individual test points

By organizing basis functions into a spatial octree hierarchy, MLFMA performs these operations at multiple scales, achieving logarithmic complexity.

---

## 2. Octree Spatial Decomposition

### 2.1 Octree Construction

MLFMA builds an octree by recursively subdividing a bounding cube into $8$ children until leaf boxes contain a small number of basis functions. The tree depth is determined by the `leaf_lambda` parameter:

```math
\text{leaf edge} = \text{leaf\_lambda}\,\lambda,\qquad
\text{nLevels} = \max\!\left(2,\left\lceil \log_2\!\left(\frac{\text{domain size}}{\text{leaf edge}}\right)\right\rceil + 1\right).
```

**Example**: For a 14λ aircraft with `leaf_lambda=1.0`:
- Leaf edge = 1.0λ
- Root edge = 16λ (smallest power of 2 containing the geometry)
- Tree depth = 5 levels (root → level 2 → 3 → 4 → leaf level 5)

### 2.2 Neighbor and Interaction Lists

At each level, boxes are classified into:
- **Neighbors**: boxes within ±1 in each $(i,j,k)$ index (including self) → near-field
- **Interaction list**: children of parent's neighbors, minus own neighbors → far-field translations

The near-field matrix $\mathbf{Z}_{\text{near}}$ stores interactions between neighboring boxes, while far-field interactions are computed via the MLFMA operators.

---

## 3. Radiation Patterns and Sampling

### 3.1 Far-Field Radiation from RWG Basis Functions

Each RWG basis function $\mathbf{f}_n$ radiates a far-field pattern. MLFMA samples this pattern on a spherical grid (Gauss-Legendre in $\theta$, uniform in $\phi$) with truncation order:

```math
L = \lfloor k d + 2.16 p^{2/3} (ke)^{1/3} \rfloor,
```

where $d = \sqrt{3} \cdot \text{edge}$ (box diagonal), $e = \text{edge}$, and $p$ is the precision parameter (default 3).

### 3.2 Four-Component Representation

The radiation pattern is stored as a 4-vector $(\mathbf{F}_{vec}, F_{scl})$:

```math
\mathbf{F}(\hat{\mathbf{k}}) = k^2 \int_S \mathbf{f}(\mathbf{r}) \, e^{-i k \hat{\mathbf{k}} \cdot \mathbf{r}} \, dS, \quad F_{scl}(\hat{\mathbf{k}}) = \frac{1}{k} \int_S (\nabla_s \cdot \mathbf{f}) \, e^{-i k \hat{\mathbf{k}} \cdot \mathbf{r}} \, dS.
```

This formulation separates the vector and scalar potentials, simplifying the EFIE computation:

```math
Z \mathbf{I} \approx -\frac{k^2 \eta_0}{16\pi^2} \sum_q w_q \left[ \mathbf{F}_{\text{rx}}^*(\hat{\mathbf{k}}_q) \cdot \mathbf{F}_{\text{tx}}(\hat{\mathbf{k}}_q) - F_{\text{rx,scl}}^*(\hat{\mathbf{k}}_q) F_{\text{tx,scl}}(\hat{\mathbf{k}}_q) \right].
```

---

## 4. Multi-Level Algorithm

### 4.1 Aggregation (Upward Pass)

Starting from the leaf level, aggregate source radiation patterns upward:
1. **Phase shift**: $e^{i k \hat{\mathbf{k}} \cdot \Delta \mathbf{r}}$ to move pattern center from child to parent
2. **Spectral interpolation**: Resample from child sampling ($L_{\text{child}}$) to parent sampling ($L_{\text{parent}}$)

**Key**: Aggregation uses **per-$m$ spectral filters** based on associated Legendre functions $P_l^m(\cos\theta)$ to correctly handle all azimuthal modes.

### 4.2 Translation (Interaction)

At each level, translate aggregated patterns between boxes in the interaction list using the translation operator:

```math
T_L(\hat{\mathbf{k}}; \mathbf{d}) = \sum_{l=0}^L (-i)^l (2l+1) \, h_l^{(2)}(k|\mathbf{d}|) \, P_l(\hat{\mathbf{k}} \cdot \hat{\mathbf{d}}),
```

where $h_l^{(2)}$ is the spherical Hankel function of the second kind and $P_l$ is the Legendre polynomial.

### 4.3 Disaggregation (Downward Pass)

Translate incoming patterns from parents to children:
1. **Phase shift**: $e^{-i k \hat{\mathbf{k}} \cdot \Delta \mathbf{r}}$ to move from parent to child center
2. **Spectral filter**: Band-limit from parent sampling to child sampling (prevents aliasing)

Again, **per-$m$ spectral filters** are essential for accuracy.

---

## 5. Per-$m$ Spectral Filters: The Correct Implementation

### 5.1 Why Separated Filters Fail

Early implementations used separated θ/φ filters: Lagrange interpolation in $\theta$ (on Gauss-Legendre nodes) and Fourier interpolation in $\phi$. This works for $m=0$ (axisymmetric modes) but **fails for $m \neq 0$** because:

```math
P_l^m(\cos\theta) \text{ for odd } m \text{ contains } \sin^m\theta \text{ factors} \Rightarrow \text{not a polynomial of } \cos\theta.
```

Lagrange interpolation on GL nodes assumes polynomial structure, causing errors when azimuthal modes couple θ and φ dependence.

### 5.2 The Correct Approach: DFT → Per-$m$ Filter → IDFT

The `_apply_disagg_filter` function implements the rigorous approach:

1. **DFT analysis in $\phi$**: Decompose data into Fourier modes $a_m, b_m$
2. **Per-$m$ θ filter**: Apply associated Legendre $P_l^m$ filter for each $m$:
   ```math
   F_{m}[i,j] = w_{\theta,j} \sum_{l=m}^{L} \frac{(2l+1)}{2} \frac{(l-m)!}{(l+m)!} P_l^m(x_i) P_l^m(x_j)
   ```
3. **IDFT synthesis**: Reconstruct at target $\phi$ points

This correctly handles all $m$ modes; in contrast, separated filters can produce significant errors when nonzero azimuthal modes dominate.

---

## 6. Accuracy and Convergence

### 6.1 Practical Verification Procedure

Accuracy is problem-dependent. A reliable workflow is:

1. Pick a smaller case where dense or ACA reference matvecs are feasible.
2. Sweep `leaf_lambda` and compare `norm(A_mlfma*x - A_ref*x) / norm(A_ref*x)`.
3. Check final observables (for example RCS cuts) in addition to raw matvec error.

### 6.2 Deep-Tree Stability Risk

Very small `leaf_lambda` values can create deep trees where high-order spherical terms become numerically delicate. If you observe instability or stagnation:

1. Increase `leaf_lambda` (fewer levels),
2. Adjust `precision`,
3. Re-validate against a trusted reference.

### 6.3 Coverage of Near/Far Decomposition

The octree neighbor and interaction-list construction is designed so each basis-function pair is treated either by near-field sparse entries or by far-field translations at some level.

---

## 7. Near-Field Preconditioning

### 7.1 Workflow Path (`solve_scattering`)

For `method=:mlfma`, the current workflow builds the preconditioner from the near-field sparse matrix stored inside the operator:

```julia
P_nf = build_nearfield_preconditioner(A_mlfma.Z_near; factorization=:ilu)
```

`preconditioner=:auto` maps to `:ilu` in this branch.

### 7.2 Optional Reordered Preconditioner

The codebase also provides `build_mlfma_preconditioner`, which reorders `Z_near` to octree ordering before ILU/LU:

```julia
P_nf = build_mlfma_preconditioner(A_mlfma; factorization=:ilu, ilu_tau=1e-2)
```

This can improve factorization behavior for some large cases and is useful for manual tuning.

---

## 8. Practical Usage

### 8.1 Basic Usage

```julia
using DiffMoM

mesh = read_obj_mesh("aircraft.obj")
freq = 3e8  # 0.3 GHz
k = 2pi * freq / 299792458.0
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))

# Auto-selects MLFMA for N > 50,000
result = solve_scattering(mesh, freq, pw)
```

### 8.2 Manual MLFMA with Preconditioner

```julia
rwg = build_rwg(mesh)
k = 2π * freq / 299792458.0

# Build MLFMA operator
A = build_mlfma_operator(mesh, rwg, k;
                          leaf_lambda=0.25,
                          verbose=true)

# Optional: build reordered-ILU preconditioner
P_nf = build_mlfma_preconditioner(A; factorization=:ilu, ilu_tau=1e-3)

# Assemble excitation and solve
v = assemble_excitation(mesh, rwg, pw)
I_coeffs, stats = solve_gmres(A, v; preconditioner=P_nf, tol=1e-6)

println("GMRES converged in $(stats.niter) iterations")
```

### 8.3 Forcing MLFMA in `solve_scattering`

```julia
# Force MLFMA even for smaller problems
result = solve_scattering(mesh, freq, pw; method=:mlfma)
```

### 8.4 Adjusting Thresholds

```julia
# Change auto-selection threshold
result = solve_scattering(mesh, freq, pw;
                          mlfma_threshold=20000)  # Use MLFMA for N > 20k
```

---

## 9. Parameter Reference

### 9.1 `build_mlfma_operator` Keywords

| Parameter | Default | Description |
|-----------|---------|-------------|
| `leaf_lambda` | 0.25 | Leaf box edge in wavelengths; tune by accuracy/time sweep for your problem. |
| `precision` | 3 | Precision parameter $p$ in truncation formula |
| `quad_order` | 3 | Quadrature order for near-field and radiation patterns |
| `verbose` | false | Print build progress information |

### 9.2 `build_mlfma_preconditioner` Keywords

| Parameter | Default | Description |
|-----------|---------|-------------|
| `factorization` | `:ilu` | One of `:ilu`, `:lu` |
| `ilu_tau` | 1e-2 | ILU drop tolerance (smaller = more fill, better preconditioner) |

### 9.3 `solve_scattering` MLFMA Path

`solve_scattering(...; method=:mlfma)` currently calls `build_mlfma_operator(mesh, rwg, k; quad_order, verbose)` and does not expose `leaf_lambda`/`precision` keywords directly. Use manual MLFMA construction when you need to tune those parameters.

---

## 10. Benchmarking Guidance

Because performance is highly geometry- and parameter-dependent, use a reproducible benchmark script for your own case. At minimum, record:

1. `N`, octree levels, and near-field `nnz` ratio,
2. operator build time,
3. preconditioner build time,
4. GMRES iterations and residual,
5. peak memory.

---

## 11. Code Mapping

| Concept | Source file | Key function / type |
|---------|------------|---------------------|
| Octree construction | `src/fast/Octree.jl` | `build_octree`, `OctreeBox`, `OctreeLevel` |
| MLFMA operator | `src/fast/MLFMA.jl` | `build_mlfma_operator`, `MLFMAOperator` |
| Radiation patterns | `src/fast/MLFMA.jl` | `compute_bf_radiation_patterns` |
| Translation factors | `src/fast/MLFMA.jl` | `compute_translation_factor` |
| Per-$m$ spectral filters | `src/fast/MLFMA.jl` | `_apply_disagg_filter`, `_build_theta_filter_m` |
| Spherical sampling | `src/fast/MLFMA.jl` | `make_sphere_sampling`, `SphereSampling` |
| Matvec | `src/fast/MLFMA.jl` | `mul!(y, A::MLFMAOperator, x)` |
| Workflow preconditioner path | `src/solver/NearFieldPreconditioner.jl` | `build_nearfield_preconditioner(A_mlfma.Z_near; ...)` |
| Optional reordered preconditioner | `src/solver/NearFieldPreconditioner.jl` | `build_mlfma_preconditioner` |
| Workflow integration | `src/Workflow.jl` | `solve_scattering(...; method=:mlfma)` |

---

## 12. Common Pitfalls

### 12.1 Using overly small `leaf_lambda`

**Problem**: Overly small `leaf_lambda` can create deep trees and unstable/slow
translations for some geometries.

**Solution**: Sweep `leaf_lambda` (for example `0.25, 0.5, 1.0`) and check
matvec or RCS error against a trusted baseline before production runs.

### 12.2 Natural-Order ILU

**Problem**: ILU quality/time can degrade on some large problems if ordering is unfavorable.

**Solution**: Try `build_mlfma_preconditioner` (reordered path) and compare against the workflow default.

### 12.3 Under-Resolved Meshes

**Problem**: MLFMA assumes smooth basis function distributions. Under-resolved meshes (edge > $\lambda/10$) violate sampling assumptions.

**Solution**: Use `solve_scattering` with `check_resolution=true` (default) to validate mesh density.

---

## 13. Adjoint and Differentiation

The MLFMA operator supports adjoint matvec via `mul!(y, A', x)`:
- Near-field: `adjoint(A.Z_near) * x`
- Far-field: Conjugated translation factors + swapped aggregation/disaggregation roles

This enables impedance optimization with MLFMA-accelerated adjoint gradient computation (see "Differentiable Design" documentation).

---

## 14. Exercises

### 14.1 Conceptual Questions

1. **Complexity**: Derive why MLFMA achieves $O(N \log N)$ complexity. Hint: count operations at each level and sum over $O(\log N)$ levels.

2. **Translation instability**: Explain why $h_l^{(2)}(kr)$ for $l > kr$ causes numerical issues. What is the physical meaning of this regime?

3. **Per-$m$ filters**: Why does separated Lagrange interpolation fail for $m \neq 0$ modes? Sketch $P_3^1(\cos\theta)$ and explain why it's not polynomial in $\cos\theta$.

### 14.2 Coding Exercises

1. Build an MLFMA operator for a 1λ-resolution sphere at 1 GHz. Print octree levels, near-field fill %, and leaf truncation order $L$.

2. Compare dense vs. MLFMA matvec accuracy for $N \approx 5000$. Use `norm(A*x - Z*x) / norm(Z*x)`.

3. Test `leaf_lambda` in `[0.75, 1.0, 1.5, 2.0]` and plot matvec error vs. number of octree levels.

4. Benchmark ILU build time with and without reordering for $N \approx 10{,}000$.

### 14.3 Advanced Challenge

Implement a frequency sweep (0.1--1.0 GHz) with MLFMA. At each frequency, rebuild the operator (geometry is constant but wavelength changes). Compare total time vs. dense assembly.

---

## 15. Chapter Checklist

- [ ] Explain MLFMA's $O(N \log N)$ complexity and why it outscales ACA H-matrices.
- [ ] Build an octree and inspect neighbor/interaction lists at each level.
- [ ] Describe aggregation, translation, and disaggregation in the multi-level algorithm.
- [ ] Understand per-$m$ spectral filters and why separated filters fail.
- [ ] Calibrate `leaf_lambda` for your geometry with a documented error/cost sweep.
- [ ] Use `build_mlfma_operator` and understand both preconditioning paths (`build_nearfield_preconditioner(A.Z_near; ...)` and optional reordered `build_mlfma_preconditioner`).
- [ ] Integrate MLFMA into `solve_scattering` with auto-selection or forced method.
- [ ] Recognize translation operator instability at deep octree levels.

---

## 16. Further Reading

- **Chew, W. C. et al.** *Fast and Efficient Algorithms in Computational Electromagnetics* (2001) -- Ch. 5-7: MLFMA theory, aggregation/disaggregation, and translation operators.
- **Ergül, Ö. & Gürel, L.** *The Multilevel Fast Multipole Algorithm (MLFMA) for Solving Large-Scale Computational Electromagnetics Problems* (2014) -- Comprehensive MLFMA reference with implementation details.
- **DiffMoM.jl source**: `src/fast/MLFMA.jl` for per-$m$ filter implementation; `src/fast/Octree.jl` for tree construction algorithms.

---

*Next: [Periodic EFIE and Floquet Metrics](07-periodic-efie-and-floquet-metrics.md) covers Ewald-periodic assembly and Floquet postprocessing for periodic unit-cell workflows.*
