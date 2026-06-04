# Periodic EFIE and Floquet Metrics

## Purpose

This chapter explains the periodic-scattering implementation added to `DiffMoM.jl`:

1. Ewald-accelerated periodic Green's correction (`PeriodicLattice`, `greens_periodic_correction`),
2. periodic EFIE assembly (`assemble_Z_efie_periodic`),
3. Floquet-mode postprocessing (`floquet_modes`, `reflection_coefficients`, `transmission_coefficients`, `power_balance`, `specular_rcs_objective`).

The focus is code-level correctness and formulation correspondence: each formula in this chapter maps directly to the implementation in `src/basis/PeriodicGreens.jl`, `src/assembly/PeriodicEFIE.jl`, and `src/postprocessing/PeriodicMetrics.jl`.

---

## Learning Goals

After this chapter, you should be able to:

1. Construct a `PeriodicLattice` and explain how Bloch wavenumbers and Ewald truncation parameters are set.
2. Write the decomposition `Z_per = Z_free + Z_corr` and identify where singular treatment is required (and where it is not).
3. Derive Floquet mode indices and classify propagating vs. evanescent orders.
4. Interpret the package's reflection/transmission amplitude conventions and power-balance modes.
5. Build a specular-direction objective matrix for periodic optimization workflows.
6. Trace each formula to concrete code lines and validation tests.

---

## 1. Periodic Lattice Model

### 1.1 Bloch Phase Parameters

For a 2D periodic lattice with periods `(dx, dy)` and incidence angles `(theta_inc, phi_inc)`, the code sets:

```math
k_x^{\text{Bloch}} = k \sin\theta_{\text{inc}} \cos\phi_{\text{inc}}, \qquad
k_y^{\text{Bloch}} = k \sin\theta_{\text{inc}} \sin\phi_{\text{inc}}.
```

This is implemented in `PeriodicLattice(dx, dy, theta_inc, phi_inc, k; ...)`.

### 1.2 Ewald Splitting and Stability Clamp

The constructor chooses:

```math
E = \max\!\left(\sqrt{\frac{\pi}{dx\,dy}}, \frac{k}{2\sqrt{\alpha_{\max}}}\right), \qquad \alpha_{\max}=2.
```

The lower bound prevents catastrophic cancellation in the Ewald decomposition (large nearly-canceling spatial/spectral terms).

### 1.3 Spectral Truncation Auto-Expansion

`N_spectral` is auto-expanded (never reduced) using the code's evanescent-margin rule (`M=5`):

```math
N_{f,x} = \left\lceil \frac{dx}{2\pi}\sqrt{k^2 + 4E^2 M^2} \right\rceil, \qquad
N_{f,y} = \left\lceil \frac{dy}{2\pi}\sqrt{k^2 + 4E^2 M^2} \right\rceil,
```

and the stored truncation is:

```math
N_f = \max(N_{\text{spectral,input}}, N_{f,x}, N_{f,y}).
```

---

## 2. Periodic Green's Correction

### 2.1 Correction Definition

The periodic kernel contribution used by the assembly is:

```math
\Delta G(\mathbf{r},\mathbf{r}') = G_{\text{per}}(\mathbf{r},\mathbf{r}') - G_0(\mathbf{r},\mathbf{r}').
```

`greens_periodic_correction(r, rp, k, lattice)` computes this using a Helmholtz-Ewald split:

1. self-correction at `(m,n)=(0,0)`,
2. spatial image sum over `(m,n) != (0,0)`,
3. Floquet/spectral sum.

Scope in current implementation:
- The periodic correction/evaluation path is used for coplanar periodic unit cells.
- Non-coplanar point offsets (`|z-z'| > 1e-12`) are rejected at runtime.
- For boundary-touching periodic conductors, assembly/postprocessing require
  Bloch-paired RWG data from `build_rwg_periodic(mesh, lattice; ...)` and reject
  non-Bloch RWG input with `ArgumentError`.

### 2.2 Wood-Anomaly Guard

In the spectral loop, modes with near-zero `|kz|` are skipped with:

```julia
abs(kz) < 1e-6 * kw && continue
```

to avoid numerical blow-up at/near Wood anomalies in this implementation path.

---

## 3. Periodic EFIE Assembly

### 3.1 Matrix Decomposition

The periodic EFIE matrix is built as:

```math
\mathbf{Z}_{\text{per}} = \mathbf{Z}_{\text{free}} + \mathbf{Z}_{\text{corr}}.
```

- `Z_free` is assembled by existing `assemble_Z_efie(...)`.
- `Z_corr` is assembled with `DeltaG`, which is smooth, so standard product quadrature is used for all triangle pairs.

### 3.2 Mixed-Potential Form in Code

For each entry:

```math
Z_{mn}^{\text{corr}} = -i\omega\mu_0 \left[
\iint \mathbf{f}_m\!\cdot\!\mathbf{f}_n \,\Delta G \,dS\,dS'
-\frac{1}{k^2}\iint (\nabla\!\cdot\!\mathbf{f}_m)(\nabla'\!\cdot\!\mathbf{f}_n)\,\Delta G \,dS\,dS'
\right].
```

with `omega_mu0 = k * eta0` in implementation.

---

## 4. Floquet Mode Enumeration

### 4.1 Mode Equations

`floquet_modes(k, lattice; N_orders=3)` enumerates:

```math
k_{x,mn} = k_x^{\text{Bloch}} + \frac{2\pi m}{dx}, \qquad
k_{y,mn} = k_y^{\text{Bloch}} + \frac{2\pi n}{dy},
```

```math
k_{z,mn}^2 = k^2 - k_{x,mn}^2 - k_{y,mn}^2.
```

- `kz2 > 0`: propagating (`propagating=true`),
- otherwise evanescent (`kz = i*sqrt(-kz2)`, `propagating=false`).

### 4.2 Stored Metadata

Each `FloquetMode` stores `(m,n,kx,ky,kz,propagating,theta_r,phi_r)` where angles are set only for propagating modes.

---

## 5. Reflection and Transmission Amplitudes

### 5.1 Reflection Coefficients

`reflection_coefficients(...)` computes current Fourier coefficients and applies:

```math
R_{mn} = -\frac{\eta_0 k}{2\,k_{z,mn}\,E_0}\,(\hat{e}_{\text{mode}} \cdot \tilde{\mathbf{J}}_{mn}).
```

where $\hat{e}_{\text{mode}}$ is the normalized projection of the incident polarization onto the mode-transverse plane.

Implementation detail:
- `R_coeffs` is allocated for all modes and aligned with `modes`,
- only propagating modes are explicitly filled; evanescent entries remain zero.

### 5.2 Transmission Convention

`transmission_coefficients(modes, R_coeffs; incident_order=(0,0))` uses:

- incident order: branch selection between `T = 1 + R` and `T = 1 - R`, choosing the lower-amplitude branch,
- non-incident orders: `T = R`.

This is the explicit convention in code for this current-sheet postprocessing path.

---

## 6. Power Balance Modes

`power_balance(...)` returns:

`(P_inc, P_refl, P_abs, P_trans, P_resid, refl_frac, abs_frac, trans_frac, resid_frac)`

with:

```math
P_{\text{inc}} = \frac{|E_0|^2 A_{\text{cell}}}{2\eta_0},
```

```math
P_{\text{refl}} = P_{\text{inc}} \sum_{\text{prop. }mn} |R_{mn}|^2\,\frac{\Re(k_{z,mn})}{k},
```

```math
P_{\text{abs}} = \frac{1}{2}\Re(\mathbf{I}^\dagger \mathbf{Z}_{\text{pen}}\mathbf{I}).
```

Transmission options in implementation:

- `transmission=:none`: `P_trans = 0`,
- `transmission=:closure`: `P_trans = clamp(P_inc - P_refl - P_abs, 0, P_inc)`,
- `transmission=:floquet`: power from `T_coeffs` (provided or inferred).

Residual is always:

```math
P_{\text{resid}} = P_{\text{inc}} - P_{\text{refl}} - P_{\text{abs}} - P_{\text{trans}}.
```

---

## 7. Specular Objective Construction

`specular_rcs_objective(...)` computes specular direction from Bloch components:

```math
\theta_{\text{spec}} = \arcsin\!\left(\frac{\sqrt{k_x^2 + k_y^2}}{k}\right), \qquad
\phi_{\text{spec}} = \operatorname{atan2}(k_y, k_x) + \pi.
```

Then:

1. builds `spec_dir`,
2. creates `mask = direction_mask(grid, spec_dir; half_angle=...)`,
3. builds `G_mat = radiation_vectors(...)`,
4. uses `pol_linear_x(grid)` for `:x/:theta/:tm` and `pol_linear_y(grid)` for `:y/:phi/:te`,
5. returns `Q = build_Q(G_mat, grid, pol; mask=mask)`.

---

## 8. Implementation Walkthrough

### 8.1 Build and Solve

```julia
lattice = PeriodicLattice(dx, dy, theta_inc, phi_inc, k)
rwg = build_rwg_periodic(mesh, lattice;
                         precheck=true, allow_boundary=true, require_closed=false)
Z_per = assemble_Z_efie_periodic(mesh, rwg, k, lattice; quad_order=3)
I = solve_forward(Z_per, v)
```

### 8.2 Floquet Metrics

```julia
modes, R = reflection_coefficients(mesh, rwg, I, k, lattice; N_orders=3)
T = transmission_coefficients(modes, R; incident_order=(0, 0))
pb = power_balance(I, Z_pen, lattice.dx * lattice.dy, k, modes, R;
                   transmission=:floquet, T_coeffs=T)
```

### 8.3 Specular Objective

```julia
Q_spec = specular_rcs_objective(mesh, rwg, grid, k, lattice;
                                half_angle=pi/18, polarization=:x)
```

---

## 9. Validation and Correspondence Checks

The implementation is directly validated by periodic test blocks in
`test/test_periodic_topology.jl`:

- Test 37: Ewald kernel behavior and convergence,
- Test 41: periodic EFIE assembly checks,
- Test 42: Floquet metrics (`floquet_modes`, `power_balance`, transmission modes).

In the project test suite, these pass under the periodic-topology block and serve as formulation-to-code regression checks.
The same block also verifies strict input validation: boundary-touching periodic meshes must use Bloch-paired RWG via `build_rwg_periodic`.

---

## 10. Code Mapping

| Concept | Source file | Key function / type |
|---------|-------------|---------------------|
| Lattice parameters and Ewald setup | `src/basis/PeriodicGreens.jl` | `PeriodicLattice` |
| Periodic Green correction | `src/basis/PeriodicGreens.jl` | `greens_periodic_correction` |
| Periodic EFIE matrix assembly | `src/assembly/PeriodicEFIE.jl` | `assemble_Z_efie_periodic` |
| Floquet mode struct/enumeration | `src/postprocessing/PeriodicMetrics.jl` | `FloquetMode`, `floquet_modes` |
| Reflection amplitudes | `src/postprocessing/PeriodicMetrics.jl` | `reflection_coefficients` |
| Transmission amplitudes | `src/postprocessing/PeriodicMetrics.jl` | `transmission_coefficients` |
| Power accounting | `src/postprocessing/PeriodicMetrics.jl` | `power_balance` |
| Specular objective matrix | `src/postprocessing/PeriodicMetrics.jl` | `specular_rcs_objective` |

---

## 11. Exercises

### Conceptual

1. Explain why `DeltaG` is smooth while `G0` is singular, and why this allows plain product quadrature for `Z_corr`.
2. Derive the propagating/evanescent criterion from `kz^2 = k^2 - kx^2 - ky^2`.
3. Compare `transmission=:closure` vs. `:floquet` in `power_balance` and explain when they should differ most.

### Coding

1. Sweep `N_orders = 1..5` and inspect how many propagating modes appear at fixed `(dx, dy, k)`.
2. For a fixed solved current `I`, evaluate `power_balance` with all transmission modes and compare `(trans_frac, resid_frac)`.
3. Build `Q_spec` for two incidence angles and verify the mask center rotates consistently with `(kx_bloch, ky_bloch)`.

---

## 12. Chapter Checklist

- [ ] Build `PeriodicLattice` from `(dx, dy, theta_inc, phi_inc, k)` and interpret its fields.
- [ ] Explain the Ewald stability clamp and spectral-order auto-expansion.
- [ ] Assemble `Z_per = Z_free + Z_corr` and identify where singular extraction is required.
- [ ] Enumerate Floquet modes and classify propagating/evanescent orders.
- [ ] Compute `R`, `T`, and `power_balance` with the correct convention for incident order.
- [ ] Build specular objective `Q_spec` and verify polarization handling.
- [ ] Trace all formulas to source files and periodic test blocks.

---

## 13. Further Reading

- Capolino, F., Wilton, D. R., & Johnson, W. A. (2005). Dynamic and static Ewald-Oseen extinction theorem for 2D periodic Green's functions. *IEEE Transactions on Antennas and Propagation*.
- Chew, W. C., Jin, J. M., Michielssen, E., & Song, J. (2001). *Fast and Efficient Algorithms in Computational Electromagnetics*.
- Balanis, C. A. (2016). *Antenna Theory* (for Floquet-mode interpretation and periodic structures).

---

*Next: [Density Topology Optimization Implementation](08-density-topology-optimization.md) derives the SIMP/filter/adjoint pipeline used by the density modules.*
