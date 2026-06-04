# Physical Optics Approximation

## Purpose

Physical Optics (PO) is a high-frequency approximation that bypasses the linear system solve entirely. Instead of assembling and inverting an impedance matrix, PO computes the surface current directly from the incident field using a simple closed-form rule: illuminated faces carry twice the tangential magnetic field, and shadow faces carry zero current. The scattered far-field is then obtained by integrating this approximate current over the mesh.

This chapter explains the PO approximation, derives its far-field integral, walks through the `solve_po` implementation in `DiffMoM.jl`, and compares PO against the full-wave MoM EFIE solver to clarify when each method is appropriate.

---

## Learning Goals

After this chapter, you should be able to:

1. State the PO approximation and its validity regime (electrically large objects, $L \gg \lambda$).
2. Derive the PO surface current from the incident plane wave fields.
3. Compute the scattered far-field by integrating PO currents over illuminated triangles.
4. Use `solve_po` to obtain bistatic RCS estimates without RWG basis functions.
5. Compare PO and MoM results, understanding where they agree and where they diverge.
6. Identify the physical phenomena that PO cannot capture (diffraction, creeping waves, multiple scattering).

---

## 1. What is Physical Optics?

### 1.1 The High-Frequency Regime

When a scattering object is electrically large -- that is, its characteristic dimension $L$ satisfies $L \gg \lambda$ where $\lambda$ is the wavelength -- the induced surface current is dominated by the local interaction between the incident wave and the surface. In this regime, each surface patch behaves approximately as an infinite tangent plane, and the current can be determined locally without solving a global linear system.

This is the Physical Optics approximation: replace the exact surface current (which requires solving the EFIE) with a locally determined current based on the tangent-plane reflection model.

### 1.2 The PO Current Rule

The PO approximation classifies every surface element into one of two categories:

- **Illuminated face**: The incident wave arrives from the outward-normal side. The PO current is:

```math
\mathbf{J}_s = 2(\hat{\mathbf{n}} \times \mathbf{H}^{\text{inc}})
```

- **Shadow face**: The incident wave arrives from behind the surface. The PO current is:

```math
\mathbf{J}_s = \mathbf{0}
```

The factor of 2 arises from the image-theory argument: on an infinite PEC plane, the reflected wave doubles the tangential magnetic field on the illuminated side.

### 1.3 What PO Captures and What It Misses

PO provides accurate results for:

- **Specular scattering**: The main reflected lobe, where the tangent-plane model is an excellent approximation.
- **Forward scattering**: The shadow boundary region in the forward direction.
- **Main-lobe RCS**: For large flat or gently curved surfaces.

PO does **not** capture:

- **Edge diffraction**: Currents near edges and discontinuities produce diffracted fields (described by GTD/UTD, not PO).
- **Creeping waves**: Currents that propagate around smooth convex surfaces into the shadow region.
- **Multiple reflections**: Re-scattering between different parts of the object (e.g., cavity interiors).
- **Resonance effects**: Low-frequency resonances that require global current solutions.

### 1.4 No Linear System Required

PO requires no matrix assembly and no linear solve -- the current is determined directly from the incident field and surface geometry. This makes PO dramatically cheaper than MoM for large objects, at the cost of reduced accuracy in non-specular directions.

---

## 2. PO Surface Current Derivation

### 2.1 Incident Plane Wave Fields

Consider a plane wave with electric field amplitude $E_0$, polarization $\hat{\mathbf{p}}$, and wave vector $\mathbf{k}$:

```math
\mathbf{E}^{\text{inc}}(\mathbf{r}) = E_0 \hat{\mathbf{p}} \, e^{-i\mathbf{k}\cdot\mathbf{r}}
```

The corresponding magnetic field is obtained from the plane-wave relation:

```math
\mathbf{H}^{\text{inc}}(\mathbf{r}) = \frac{1}{\eta_0} \hat{\mathbf{k}} \times \mathbf{E}^{\text{inc}}(\mathbf{r}) = \frac{E_0}{\eta_0} (\hat{\mathbf{k}} \times \hat{\mathbf{p}}) \, e^{-i\mathbf{k}\cdot\mathbf{r}}
```

where $\eta_0 \approx 376.73\;\Omega$ is the free-space impedance and $\hat{\mathbf{k}} = \mathbf{k}/|\mathbf{k}|$ is the propagation direction unit vector.

### 2.2 Illumination Test

A triangle with outward unit normal $\hat{\mathbf{n}}_t$ is illuminated when the incident wave arrives from the normal side. In the implementation, this is:

```math
\hat{\mathbf{k}} \cdot \hat{\mathbf{n}}_t \le 0
```

The wave propagation direction $\hat{\mathbf{k}}$ points toward the surface when the angle between $\hat{\mathbf{k}}$ and $\hat{\mathbf{n}}_t$ is obtuse. Triangles where $\hat{\mathbf{k}} \cdot \hat{\mathbf{n}}_t > 0$ are in shadow and carry no PO current.

### 2.3 PO Current on an Illuminated Triangle

Substituting the plane wave magnetic field into the PO rule:

```math
\mathbf{J}_s(\mathbf{r}) = 2(\hat{\mathbf{n}}_t \times \mathbf{H}^{\text{inc}}) = \frac{2E_0}{\eta_0} \hat{\mathbf{n}}_t \times (\hat{\mathbf{k}} \times \hat{\mathbf{p}}) \, e^{-i\mathbf{k}\cdot\mathbf{r}}
```

Define the **vector factor** for each triangle:

```math
\mathbf{V}_t = \hat{\mathbf{n}}_t \times (\hat{\mathbf{k}} \times \hat{\mathbf{p}})
```

This vector is constant over the triangle (the normal $\hat{\mathbf{n}}_t$ is constant on a flat triangle, and $\hat{\mathbf{k}}$, $\hat{\mathbf{p}}$ are global constants). Only the exponential phase factor $e^{-i\mathbf{k}\cdot\mathbf{r}}$ varies spatially across the triangle.

---

## 3. Far-Field Integration

### 3.1 Scattered Far-Field Expression

The scattered far-field from a surface current $\mathbf{J}_s$ is given by the radiation integral:

```math
\mathbf{E}^{\text{scat}}(\hat{\mathbf{r}}) = \frac{-ik\eta_0}{4\pi} \int_{\Gamma_{\text{illum}}} \hat{\mathbf{r}} \times (\hat{\mathbf{r}} \times \mathbf{J}_s(\mathbf{r}')) \, e^{ik\hat{\mathbf{r}}\cdot\mathbf{r}'} \, dS'
```

where $\hat{\mathbf{r}}$ is the observation direction, $k$ is the wavenumber, and the integration runs over the illuminated portion of the surface $\Gamma_{\text{illum}}$.

### 3.2 Prefactor Simplification

Substituting the PO current expression:

```math
\mathbf{E}^{\text{scat}}(\hat{\mathbf{r}}) = \frac{-ik\eta_0}{4\pi} \cdot \frac{2E_0}{\eta_0} \sum_t \left[\hat{\mathbf{r}} \times (\hat{\mathbf{r}} \times \mathbf{V}_t)\right] \int_{T_t} e^{ik(\hat{\mathbf{r}} - \hat{\mathbf{k}})\cdot\mathbf{r}'} \, dS'
```

The prefactors simplify:

```math
\frac{-ik\eta_0}{4\pi} \cdot \frac{2E_0}{\eta_0} = \frac{-ikE_0}{2\pi}
```

This cancellation of $\eta_0$ is a useful simplification in the implementation.

### 3.3 Per-Triangle Phase Integral

For each illuminated triangle $T_t$ with area $A_t$, the code evaluates

```math
I_t = \int_{T_t}\exp\!\left(ik(\hat{\mathbf{r}}-\hat{\mathbf{k}})\cdot \mathbf{r}'\right)\,dS'
```

using the analytical triangle formula implemented in `_phase_integral_analytical(...)` (POFacets-style with stable special-case branches for small phase differences), not numerical quadrature.

### 3.4 Vector Cross-Product Identity

The double cross product simplifies via the BAC-CAB identity:

```math
\hat{\mathbf{r}} \times (\hat{\mathbf{r}} \times \mathbf{V}_t) = (\hat{\mathbf{r}} \cdot \mathbf{V}_t)\hat{\mathbf{r}} - \mathbf{V}_t
```

This projects out the radial component of $\mathbf{V}_t$, keeping only the transverse (radiating) component. Physically, far-field radiation is always transverse to the observation direction.

### 3.5 Complete Far-Field Formula

Combining all pieces, the scattered far-field at observation direction $\hat{\mathbf{r}}$ is:

```math
\mathbf{E}^{\text{scat}}(\hat{\mathbf{r}}) = \frac{-ikE_0}{2\pi} \sum_{t \in \text{illum}} \left[(\hat{\mathbf{r}} \cdot \mathbf{V}_t)\hat{\mathbf{r}} - \mathbf{V}_t\right] I_t
```

This is a direct summation over illuminated triangles -- no matrix inversion is needed.

---

## 4. Implementation in DiffMoM.jl

### 4.1 The `solve_po` Function

The PO solver is implemented as a single function with the following signature:

```julia
solve_po(mesh::TriMesh, freq_hz::Real, excitation::PlaneWaveExcitation;
         grid::SphGrid = make_sph_grid(36, 72),
         c0::Float64 = 299792458.0,
         eta0::Float64 = 376.730313668)
```

**Key inputs:**

- `mesh::TriMesh` -- a triangle mesh of the scatterer. No RWG basis functions are needed.
- `freq_hz` -- operating frequency in Hz.
- `excitation::PlaneWaveExcitation` -- the incident plane wave (wave vector, amplitude, polarization).
- `grid::SphGrid` -- spherical observation grid for far-field evaluation.

### 4.2 The `POResult` Struct

The function returns a `POResult` containing all relevant output:

```julia
struct POResult
    E_ff::Matrix{ComplexF64}     # (3, N_omega) scattered far-field
    J_s::Vector{CVec3}           # PO surface current per triangle centroid
    illuminated::BitVector       # which triangles are illuminated
    grid::SphGrid                # observation grid
    freq_hz::Float64             # frequency
    k::Float64                   # wavenumber
end
```

The fields store the scattered far-field (3 Cartesian components per observation direction), the PO current at each triangle centroid (for visualization), and a bit-vector marking illuminated triangles.

### 4.3 Phase 1: Illumination and Current Computation

The first phase loops over all triangles, tests $\hat{\mathbf{k}} \cdot \hat{\mathbf{n}} \le 0$, and computes $\mathbf{V}_t$ for illuminated triangles. Shadow triangles get zero current and zero vector factor. The centroid current `J_s[t]` is stored for diagnostics.

### 4.4 Phase 2: Far-Field Integration

The second phase is a double loop over observation directions ($N_\Omega$) and illuminated triangles ($N_t$). For each pair, it computes the analytical phase integral `I_t`, applies the transverse projection $(\hat{\mathbf{r}} \cdot \mathbf{V}_t)\hat{\mathbf{r}} - \mathbf{V}_t$, and accumulates with the prefactor $-ikE_0/(2\pi)$:

```julia
prefactor = -1im * k * E0 / (2pi)

for q in 1:N_omega
    r_hat = Vec3(grid.rhat[:, q])
    delta_k = r_hat - k_hat       # combined phase direction

    for t in 1:Nt
        !illuminated[t] && continue
        I_t = _phase_integral_analytical(
            k, delta_k, tri_v1[t], tri_v2[t], tri_v3[t], tri_area[t]
        )
        proj = r_hat * dot(r_hat, V_t[t]) - V_t[t]
        E_q += CVec3(complex.(prefactor * proj * I_t))
    end
end
```

### 4.5 Computational Complexity

The PO solver has complexity $O(N_t \times N_\Omega)$, where $N_t$ is the number of illuminated triangles and $N_\Omega$ is the number of observation directions. There is no $O(N^2)$ matrix assembly and no $O(N^3)$ factorization.

---

## 5. PO vs. MoM Comparison

### 5.1 Side-by-Side Feature Comparison

| Aspect | MoM (EFIE) | Physical Optics |
|--------|-----------|-----------------|
| Accuracy | Full-wave, exact for given mesh resolution | High-frequency approximation, no diffraction |
| Complexity | $O(N^2)$ assembly + $O(N^3)$ solve | $O(N_t \times N_\Omega)$ direct evaluation |
| Frequency range | Low to moderate (mesh-limited) | High frequency (electrically large objects) |
| Edge effects | Captured via RWG currents | Missing (no edge diffraction) |
| Multiple scattering | Included through full coupling | Not included |
| Creeping waves | Captured (with sufficient mesh) | Not captured |
| Basis functions | RWG required | Not needed (works on raw mesh) |
| Use case | Validation reference, optimization | Fast RCS estimate, MoM validation |

### 5.2 When to Use Each Method

**Use MoM when:**
- The object is electrically small to moderate ($L \lesssim 10\lambda$).
- You need accurate results in all scattering directions, including shadow regions.
- Edge diffraction, creeping waves, or multiple reflections are important.
- You are performing gradient-based impedance optimization (PO does not support impedance parameters).

**Use PO when:**
- The object is electrically large ($L \gg \lambda$) and MoM is computationally prohibitive.
- You only need the specular/near-specular RCS pattern.
- You want a fast independent reference to validate MoM results at high frequencies.
- You need a quick estimate before committing to a full MoM simulation.

---

## 6. MoM vs. PO Validation Workflow

At high frequencies where both methods are applicable, PO provides an independent reference for validating MoM results: run both on the same mesh and frequency, compare bistatic RCS, expect agreement in specular/forward regions, and expect disagreement near shadow boundaries where diffraction (captured by MoM, missed by PO) dominates.

### 6.1 Complete Example

```julia
using DiffMoM

# Set up the problem
mesh = read_obj_mesh("plate.obj")
freq = 10e9
c0 = 299792458.0
k = 2pi * freq / c0
pw = make_plane_wave(Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
grid = make_sph_grid(91, 361)

# --- PO solution (fast, approximate) ---
po_result = solve_po(mesh, freq, pw; grid=grid)

# --- MoM solution (slower, full-wave) ---
rwg = build_rwg(mesh)
Z = assemble_Z_efie(mesh, rwg, k)
V = assemble_excitation(mesh, rwg, pw)
I_coeffs = Z \ V
G_mat = radiation_vectors(mesh, rwg, grid, k)
E_ff_mom = compute_farfield(G_mat, I_coeffs, length(grid.w))

# --- Compare bistatic RCS ---
rcs_po  = bistatic_rcs(po_result.E_ff; E0=1.0)
rcs_mom = bistatic_rcs(E_ff_mom; E0=1.0)
```

### 6.2 Interpreting Differences

When comparing PO and MoM RCS patterns, the specular peak should agree within 1--2 dB for electrically large surfaces, and main-lobe width should be similar. First sidelobes may differ slightly due to edge effects. The largest discrepancies appear in the shadow region (PO predicts near-zero RCS; MoM shows diffraction) and at edge-on directions (PO misses edge-diffracted fields entirely). These differences reveal the relative importance of specular vs. diffraction contributions.

---

## 7. Code Mapping

### 7.1 Source Files

| File | Role |
|------|------|
| `src/postprocessing/PhysicalOptics.jl` | PO solver: `solve_po`, `POResult` struct |
| `src/geometry/Mesh.jl` | Triangle mesh I/O: `read_obj_mesh`, `triangle_normal`, `triangle_area`, `triangle_center` |
| `src/assembly/Excitation.jl` | Plane wave construction: `make_plane_wave`, `PlaneWaveExcitation` |
| `src/postprocessing/FarField.jl` | Far-field utilities: `compute_farfield`, `radiation_vectors`, `SphGrid`, `make_sph_grid` |
| `src/postprocessing/Diagnostics.jl` | RCS utilities: `bistatic_rcs` |
| `src/Types.jl` | Vector types: `Vec3`, `CVec3` |

### 7.2 Key Data Flow

`PlaneWaveExcitation` + `TriMesh` feed into `solve_po`, which runs Phase 1 (illumination test, $\mathbf{V}_t$ computation) and Phase 2 (far-field integration over $N_\Omega \times N_t$ pairs), producing a `POResult`. Note that `solve_po` does **not** call `build_rwg`, `assemble_Z_efie`, or any solve routine -- it operates directly on the triangle mesh.

---

## 8. Exercises

### 8.1 Conceptual Questions

1. **Illumination geometry**: For a sphere illuminated by a plane wave propagating in the $-\hat{\mathbf{z}}$ direction, sketch which triangles are illuminated and which are in shadow. What fraction of the total surface area is illuminated?

2. **PO current discontinuity**: The PO approximation produces a discontinuous current at the shadow boundary (abrupt jump from $2\hat{\mathbf{n}} \times \mathbf{H}^{\text{inc}}$ to zero). Explain why this discontinuity is unphysical, and what it implies about the accuracy of PO near the shadow boundary.

3. **Frequency scaling**: How does the number of triangles needed for PO scale with frequency, and why is PO still practical at high frequencies even though the mesh must resolve the geometry?

### 8.2 Numerical Experiments

4. **Flat plate RCS**: Use `solve_po` and `solve_scattering` (MoM) to compute the bistatic RCS of a $1\lambda \times 1\lambda$ flat plate at broadside incidence. Compare the two results and identify the angular regions of agreement and disagreement.

```julia
using DiffMoM

freq = 3e9
c0 = 299792458.0
lam = c0 / freq
mesh = make_rect_plate(lam, lam, 10, 10)
k = 2pi * freq / c0
pw = make_plane_wave(Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
grid = make_sph_grid(181, 1)  # phi=0 cut

po_result = solve_po(mesh, freq, pw; grid=grid)
# Compare with MoM result...
```

5. **Angular-grid convergence**: Run `solve_po` with progressively finer observation grids (for example `make_sph_grid(31, 61)`, `make_sph_grid(61, 121)`, `make_sph_grid(121, 241)`). Compare bistatic-RCS cuts and identify when the angular sampling is converged.

6. **Illumination fraction**: For a sphere mesh, compute the fraction of triangles marked as illuminated by `solve_po`. Verify that it approaches 50% as the mesh becomes finer.

### 8.3 Advanced

7. **PO + PTD extension**: Research the Physical Theory of Diffraction (PTD), which adds edge-diffraction corrections to PO. Outline how you would implement a PTD correction on top of `solve_po`.

8. **Multiple-bounce PO**: For a dihedral corner reflector, describe an algorithm that iteratively applies PO (reflected field from the first bounce becomes the incident field for the second bounce).

---

## 9. Chapter Checklist

Before proceeding, ensure you understand:

- [ ] The PO approximation: $\mathbf{J}_s = 2(\hat{\mathbf{n}} \times \mathbf{H}^{\text{inc}})$ on illuminated faces, $\mathbf{J}_s = \mathbf{0}$ on shadow faces.
- [ ] The illumination test used in code: a triangle is illuminated when $\hat{\mathbf{k}} \cdot \hat{\mathbf{n}} \le 0$.
- [ ] The far-field radiation integral and its prefactor simplification to $-ikE_0/(2\pi)$.
- [ ] The per-triangle phase integral $I_t$ is evaluated analytically by `_phase_integral_analytical`.
- [ ] The vector identity $\hat{\mathbf{r}} \times (\hat{\mathbf{r}} \times \mathbf{V}) = (\hat{\mathbf{r}} \cdot \mathbf{V})\hat{\mathbf{r}} - \mathbf{V}$.
- [ ] That PO requires no RWG basis functions and no linear system solve.
- [ ] The computational complexity: $O(N_t \times N_\Omega)$ versus MoM's $O(N^2)$ assembly + $O(N^3)$ solve.
- [ ] What PO misses: edge diffraction, creeping waves, multiple reflections, resonances.
- [ ] How to use `solve_po` and `POResult` in `DiffMoM.jl`.
- [ ] The cross-validation workflow: comparing PO and MoM RCS to identify specular vs. diffraction contributions.

---

## 10. Further Reading

1. **Physical Optics and high-frequency methods:**
   - Balanis, C. A. (2012). *Advanced Engineering Electromagnetics*, 2nd ed. Wiley. Chapters 11--12 cover PO and diffraction theory.
   - Knott, E. F., Shaeffer, J. F., & Tuley, M. T. (2004). *Radar Cross Section*, 2nd ed. SciTech Publishing. Comprehensive treatment of PO for RCS prediction.

2. **Physical Theory of Diffraction (PTD):**
   - Ufimtsev, P. Ya. (2007). *Fundamentals of the Physical Theory of Diffraction*. Wiley. The definitive reference on edge-diffraction corrections to PO.

3. **Comparison of asymptotic and full-wave methods:**
   - Gibson, W. C. (2008). *The Method of Moments in Electromagnetics*. Chapman & Hall/CRC. Chapter 10 discusses PO and its relationship to MoM.

---

*Next: [Matrix-Free EFIE Operators](04-matrix-free-operators.md) introduces the memory-efficient matrix-free operator infrastructure for iterative solves without dense matrix storage.*
