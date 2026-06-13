# 3D Volume Material Scattering (DDA / VIE)

## Purpose

The surface EFIE path solves for currents on the *boundary* of a perfect
conductor. But many problems involve penetrable material that fills a *volume*:
dielectric resonators, magnetodielectric absorbers, gradient-index lenses,
bianisotropic metamaterials, or a topology-optimized permittivity distribution.
For these, the unknown is the field *inside* the material, and the natural
discretization fills the scatterer with a uniform grid of small voxels rather
than meshing a surface.

This chapter develops the 3D volume scattering subsystem in `DiffMoM.jl`, which
uses a vector discrete-dipole approximation (DDA) -- equivalently a
volume-integral-equation (VIE) discretization with one polarizable cell per
voxel. We derive the coupled-dipole system from the volume polarization, build
the free-space dipole dyadic and the Clausius-Mossotti polarizability, assemble
and solve the linear system (dense, matrix-free, or FFT-accelerated), extract
induced dipoles and far fields, and validate against the exact dielectric-Mie
solution in the Rayleigh limit. We also cover the coupled electric-magnetic
(magnetodielectric / bianisotropic) extension and the adjoint sensitivity used
for permittivity design.

---

## Learning Goals

After this chapter, you should be able to:

1. State the time/phase convention (`exp(+i omega t)`, scalar Green phase `exp(-i k R)`) and why every sign in the subsystem follows from it.
2. Derive the coupled-dipole (VIE) governing system from the volume polarization current.
3. Construct the free-space electric dipole dyadic $\mathbf{G}_{EE}$ and explain why the self term is excluded.
4. Compute the normalized Clausius-Mossotti polarizability and (optionally) the radiative correction.
5. Excite the system with a transverse plane wave and solve for the total field via dense LU, matrix-free GMRES, or FFT-accelerated GMRES.
6. Post-process induced dipoles, scattered near fields, far-field amplitude, and bistatic RCS.
7. Extend to magnetodielectric and bianisotropic material via the coupled `6 x 6` EM DDA.
8. Compute permittivity design gradients with the DDA adjoint.
9. Choose the right operator (dense / matrix-free / FFT) for a given voxel count.

---

## 1. Time Convention and the Free-Space Green Function

The entire 3D volume subsystem uses the engineering time convention
$e^{+i\omega t}$, identical to the rest of `DiffMoM.jl`. This single choice
fixes every sign that follows. With $e^{+i\omega t}$, an outgoing spherical wave
carries the phase $e^{-ikR}$, so the scalar free-space Green function is

```math
g(R) = \frac{e^{-ikR}}{4\pi R}, \qquad R = |\mathbf{r} - \mathbf{r}'|.
```

This is stated in the file headers (`src/mom3d/DDA3D.jl`,
`src/mom3d/EMDDA3D.jl`) and realized directly in the dipole dyadic, where the
common prefactor is `expfac = exp(-1im*k*R)/(4pi)`. Three downstream signs all
follow from this convention and are worth fixing in your mind now:

- **Plane wave**: $\mathbf{E}^{\text{inc}}(\mathbf{r}) = \hat{\mathbf{p}}\,E_0\,e^{-i\mathbf{k}\cdot\mathbf{r}}$ (negative exponent).
- **Far field**: $\mathbf{E}^{\text{scat}}(\mathbf{r}) \sim \dfrac{e^{-ikr}}{r}\,\mathbf{F}(\hat{\mathbf{r}})$, with the conjugate phase $e^{+ik\,\hat{\mathbf{r}}\cdot\mathbf{r}_j}$ appearing *inside* the dipole sum.
- **Radiative correction**: the denominator $1 + i k^3 \alpha_0/(6\pi)$ (the sign of $i$ is the $e^{+i\omega t}$-consistent form).

If you ever port a formula from a reference that uses the physics convention
$e^{-i\omega t}$, conjugate it (flip the sign of every $i$) before comparing.

---

## 2. From Volume Polarization to the Coupled-Dipole System

### 2.1 The Volume Equivalence

A penetrable scatterer of relative permittivity $\varepsilon_r(\mathbf{r})$ in
free space supports a polarization current. In the volume-integral picture, the
total electric field anywhere is the incident field plus the field radiated by
the induced polarization throughout the material:

```math
\mathbf{E}(\mathbf{r}) = \mathbf{E}^{\text{inc}}(\mathbf{r})
   + \int_V \mathbf{G}_{EE}(\mathbf{r}, \mathbf{r}')\,
            \frac{\mathbf{p}(\mathbf{r}')}{\varepsilon_0}\, dV'.
```

Here $\mathbf{p}/\varepsilon_0$ is the *normalized* dipole density and
$\mathbf{G}_{EE}$ is the free-space electric dipole dyadic (Section 3) that maps a
normalized dipole $\mathbf{q} = \mathbf{p}/\varepsilon_0$ to the electric field it
radiates.

### 2.2 Discretization into Voxels

Discretize the volume into $N$ uniform voxels of volume $V$ centered at
$\mathbf{r}_i$. Each voxel is replaced by a single point dipole at its center
whose normalized moment is proportional to the local field through a
polarizability $\alpha_j$ (Section 4):

```math
\mathbf{q}_j = \alpha_j\, \mathbf{E}(\mathbf{r}_j).
```

Evaluating the integral equation at each voxel center, replacing the integral by
a sum over the *other* voxels (the self term is folded into $\alpha$), gives the
coupled-dipole / VIE governing system:

```math
\mathbf{E}_i - \sum_{j \neq i} \mathbf{G}_{EE}(\mathbf{r}_i, \mathbf{r}_j)\,
   \alpha_j\, \mathbf{E}_j = \mathbf{E}^{\text{inc}}_i,
\qquad i = 1, \dots, N.
```

The unknowns are the **total** electric fields $\mathbf{E}_j$ at the voxel
centers (three complex components each, so $3N$ scalar unknowns). In operator
form the system matrix has the block structure

```math
A_{ij} = \delta_{ij}\,\mathbf{I}_3 - \mathbf{G}_{EE}(\mathbf{r}_i, \mathbf{r}_j)\,\alpha_j,
```

i.e. identity on the diagonal minus the interaction blocks off-diagonal. This is
exactly the convention used throughout (`assemble_dda_3d`, the matrix-free
`DDAOperator3D` matvec, and the adjoint). Once the total fields are known, the
induced dipoles and all observables follow by direct evaluation.

> **Note.** Because the unknown is the *total* field at each center and the self
> interaction is absorbed into $\alpha$, a vacuum voxel ($\varepsilon_r = 1$)
> gets $\alpha = 0$ and contributes nothing. You can therefore carve an
> arbitrary shape out of a box grid simply by setting $\varepsilon_r = 1$
> outside it.

---

## 3. The Free-Space Electric Dipole Dyadic

The coupling block $\mathbf{G}_{EE}(\mathbf{r}, \mathbf{r}')$ is the field at
$\mathbf{r}$ produced by a normalized dipole $\mathbf{q} = \mathbf{p}/\varepsilon_0$
at $\mathbf{r}'$. With $\mathbf{R} = \mathbf{r} - \mathbf{r}'$, $R = |\mathbf{R}|$,
and $\hat{\mathbf{R}} = \mathbf{R}/R$,

```math
\mathbf{G}_{EE}(\mathbf{r}, \mathbf{r}') =
   \frac{e^{-ikR}}{4\pi}\left[
     \frac{k^2}{R}\left(\mathbf{I} - \hat{\mathbf{R}}\hat{\mathbf{R}}^{\!\top}\right)
     + \left(\frac{1}{R^3} + \frac{ik}{R^2}\right)
       \left(3\,\hat{\mathbf{R}}\hat{\mathbf{R}}^{\!\top} - \mathbf{I}\right)
   \right].
```

The two terms have a clear physical reading:

- The $k^2/R$ term is the **radiation (far) term**: transverse to $\hat{\mathbf{R}}$ via the projector $\mathbf{I} - \hat{\mathbf{R}}\hat{\mathbf{R}}^{\!\top}$, and dominant at large $R$.
- The $(1/R^3 + ik/R^2)$ term is the **near/induction term**, with the static dipole shape $3\hat{\mathbf{R}}\hat{\mathbf{R}}^{\!\top} - \mathbf{I}$.

This dyadic is `electric_dipole_dyadic_3d(r, rp, k0)`. It is **singular at
$\mathbf{r} = \mathbf{r}'$** ($R = 0$) and raises an error there; the self
interaction is *not* evaluated by this function but is instead represented by the
polarizability model. As a consequence, observation points passed to
`scattered_field_dda_3d` must never coincide with a voxel center.

The dyadic is also reciprocal: the off-diagonal interaction block satisfies
$\mathbf{G}_{EE}(\mathbf{r}_i, \mathbf{r}_j) = \mathbf{G}_{EE}(\mathbf{r}_j, \mathbf{r}_i)^{\!\top}$,
a symmetry verified in the test suite to $< 10^{-13}$.

---

## 4. Polarizability: Clausius-Mossotti

The polarizability $\alpha_j$ closes the system by relating each voxel's induced
dipole to its local total field. The package uses the **normalized** electric
polarizability $\alpha = \mathbf{p}/(\varepsilon_0 \mathbf{E})$, which has units
of volume (m$^3$), so the induced normalized dipole is simply
$\mathbf{q} = \alpha\,\mathbf{E}$.

### 4.1 The Clausius-Mossotti Form

For an isotropic voxel of relative permittivity $\varepsilon_r$ and volume $V$,

```math
\alpha_0 = 3V\,\frac{\varepsilon_r - 1}{\varepsilon_r + 2}.
```

This is the **exact electrostatic polarizability of a sphere of the same
volume**. For an anisotropic tensor permittivity $\boldsymbol{\varepsilon}_m$ the
tensor generalization is

```math
\boldsymbol{\alpha}_0 = 3V\,(\boldsymbol{\varepsilon}_m - \mathbf{I})\,(\boldsymbol{\varepsilon}_m + 2\mathbf{I})^{-1}.
```

Both are implemented in `clausius_mossotti_polarizability(eps_r, volume; ...)`
(scalar and `AbstractMatrix` methods). The denominator is singular near
$\varepsilon_r = -2$ (or $\det(\boldsymbol{\varepsilon}_m + 2\mathbf{I}) \approx 0$);
the code raises an error in that regime.

> **The $3V$ vs $4\pi a^3$ subtlety.** The normalized polarizability uses the
> volume coefficient $3V$. The single-sphere Rayleigh polarizability used as a
> *validation reference* is written $4\pi a^3 (\varepsilon_r - 1)/(\varepsilon_r + 2)$
> in terms of the sphere radius $a$. These agree only in the low-frequency limit
> as the voxelization is refined; at a finite voxel count the staircased sphere
> is not exactly the smooth sphere, so expect $\sim 0.5$--$1\%$ relative error at
> a $7^3$ grid (Section 8).

### 4.2 The Radiative Correction (Optional, Off by Default)

The bare electrostatic $\alpha_0$ does not conserve energy at finite frequency.
The leading-order radiation-reaction correction, in the $e^{+i\omega t}$
convention, is

```math
\alpha = \frac{\alpha_0}{1 + i\,k^3 \alpha_0 / (6\pi)}
\qquad\text{(scalar)},
\qquad
\boldsymbol{\alpha} = \boldsymbol{\alpha}_0\left(\mathbf{I} + i\,k^3 \boldsymbol{\alpha}_0 / (6\pi)\right)^{-1}
\qquad\text{(tensor)}.
```

Enable it with `radiative_correction=true`. It is **off by default** because the
low-frequency validation targets the bare Clausius-Mossotti limit, and because
the permittivity-design adjoint (Section 7) only supports the uncorrected form.

`dda_polarizabilities(grid, k0, eps_r; radiative_correction=false)` applies the
chosen form to every voxel, returning one scalar or `3 x 3` tensor per voxel.

---

## 5. Excitation, Solve, and Observables

### 5.1 Plane-Wave Excitation

The incident field is sampled at the voxel centers by
`planewave_dda_3d(grid, k_vec, E0, pol)`:

```math
\mathbf{E}^{\text{inc}}(\mathbf{r}) = \hat{\mathbf{p}}\,E_0\,e^{-i\mathbf{k}\cdot\mathbf{r}},
```

where `k_vec` $= k\,\hat{\mathbf{k}}$. The polarization must be **transverse** to
the wave vector: the code enforces $|\hat{\mathbf{k}}\cdot\hat{\mathbf{p}}| \le 10^{-10}$
and errors otherwise (both must be nonzero). The negative exponent matches the
$e^{-ikR}$ convention from Section 1.

### 5.2 Solving the System

`solve_dda_3d(grid, k0, eps_r, E_inc; solver=...)` returns a `DDAResult3D`
holding the total fields, polarizabilities, and solver metadata. Two solver
modes are available:

| `solver` | Operator | Cost | Use when |
|----------|----------|------|----------|
| `:direct` | dense `3N x 3N` via `assemble_dda_3d` + LU | $O(N^2)$ storage, $O(N^3)$ factor | tiny grids; reusing the LU for the adjoint |
| `:gmres` | matrix-free `DDAOperator3D` | $O(N^2)$/matvec, near-zero alloc | any non-tiny grid |

For larger grids, build `fft_dda_operator_3d(grid, k0, eps_r)` and drive it with
a Krylov solver directly: on a uniform grid the dipole-interaction matrix is
block-Toeplitz, so the dense all-pairs sum becomes a zero-padded convolution
(padded dims $(2n_x-1, 2n_y-1, 2n_z-1)$) evaluated by FFT, giving $O(N\log N)$
matvecs that match the dense operator to machine precision (Section 6).

### 5.3 Induced Dipoles and Fields

From a solved `DDAResult3D`:

- `induced_dipoles_dda_3d(res)` returns $\mathbf{q}_j = \alpha_j \mathbf{E}_j$ per voxel. Vacuum voxels yield zero.
- `scattered_field_dda_3d(res, r_obs)` sums the radiated near field of all induced dipoles at observation points (which must not coincide with voxel centers).
- `farfield_dda_3d(res, rhat)` returns the far-field amplitude.

The far-field amplitude $\mathbf{F}(\hat{\mathbf{r}})$, defined by
$\mathbf{E}^{\text{scat}}(\mathbf{r}) \sim \dfrac{e^{-ikr}}{r}\mathbf{F}(\hat{\mathbf{r}})$,
is the transverse projection of the coherently phased dipole sum:

```math
\mathbf{F}(\hat{\mathbf{r}}) = \frac{k_0^2}{4\pi}
   \sum_j e^{+i k_0\,\hat{\mathbf{r}}\cdot\mathbf{r}_j}\,
   \left(\mathbf{I} - \hat{\mathbf{r}}\hat{\mathbf{r}}^{\!\top}\right)\mathbf{q}_j.
```

The conjugate phase $e^{+ik_0\hat{\mathbf{r}}\cdot\mathbf{r}_j}$ is consistent
with the $e^{-ikr}$ radial factor pulled out front. The bistatic radar cross
section follows from the amplitude:

```math
\sigma(\hat{\mathbf{r}}) = 4\pi\,\Re\!\left(\mathbf{F}\cdot\mathbf{F}\right).
```

---

## 6. FFT Acceleration

A uniform Cartesian grid makes the dipole-interaction operator translationally
invariant: the block $\mathbf{G}_{EE}(\mathbf{r}_i, \mathbf{r}_j)$ depends only on
the integer offset $\mathbf{r}_i - \mathbf{r}_j$. The operator is therefore
**block-Toeplitz**, and the matvec $\mathbf{y} = \mathbf{x} - \mathbf{G}\,(\alpha\!\cdot\!\mathbf{x})$
(with the singular self offset excluded) is a discrete convolution.

Embedding the Toeplitz operator in a circulant one of padded size
$(2n_x-1, 2n_y-1, 2n_z-1)$ turns each matvec into three forward FFTs, a
pointwise multiply by the precomputed kernel spectrum, and an inverse FFT --
reducing the per-matvec cost from $O(N^2)$ to $O(N\log N)$. The result is
identical to the dense operator to machine precision (verified to $< 10^{-12}$ in
the FFT tests).

- `fft_dda_operator_3d(grid, k0, eps_r)` -- electric path; drive with a Krylov solver.
- `fft_em_dda_operator_3d(grid, k0, eps_r, mu_r)` -- coupled EM path; also wired into `solve_em_dda_3d(...; solver=:fft_gmres)`.

FFT operators require a uniform grid (the source of the Toeplitz structure) and
a strictly positive `k0`.

---

## 7. Coupled Electric-Magnetic and Bianisotropic DDA

When the material is magnetic ($\mu_r \neq 1$) or bianisotropic, electric and
magnetic dipoles couple and we need six unknowns per voxel, ordered
$(E_x, E_y, E_z, H_x, H_y, H_z)$. Each voxel carries a `6 x 6` polarizability
mapping $[\mathbf{E}; \mathbf{H}] \to [\mathbf{q}; \mathbf{m}]$.

### 7.1 Magnetodielectric Voxels

The magnetic dipole uses the *same* Clausius-Mossotti form with $\mu_r$ in place
of $\varepsilon_r$, so $\mathbf{m} = \alpha_m \mathbf{H}$ with
$\alpha_m = 3V(\mu_r - 1)/(\mu_r + 2)$
(`magnetic_clausius_mossotti_polarizability`). The cross terms come from the
electromagnetic dual of the electric dyadic. With $G = e^{-ikR}/(4\pi R)$ and the
gradient factor $\partial_R G = (-ik - 1/R)\,G$,

```math
\mathbf{E}\ \text{from a magnetic dipole } \mathbf{m}: \quad -i\,\eta_0 k\,(\nabla G \times \mathbf{m}),
\qquad
\mathbf{H}\ \text{from an electric dipole } \mathbf{q}: \quad \frac{i k}{\eta_0}\,(\nabla G \times \mathbf{q}),
```

with the free-space impedance $\eta_0 = 376.730313668\ \Omega$.

Use `solve_em_dda_3d(grid, k0, eps_r, mu_r, E_inc, H_inc; solver=...)` with the
incident pair from `planewave_em_dda_3d` (which sets
$\mathbf{H} = \hat{\mathbf{k}}\times\mathbf{E}/\eta_0$). The far field
`farfield_em_dda_3d` returns $(\mathbf{F}_E, \mathbf{F}_H)$ that satisfy the deep
far-zone radiation condition $\mathbf{F}_E = -\eta_0\,(\hat{\mathbf{n}}\times\mathbf{F}_H)$.

### 7.2 Bianisotropic Constitutive Mapping

For a general bianisotropic medium, supply a normalized `6 x 6` constitutive
matrix $\mathsf{C}_6$ that acts on $[\mathbf{E}; \eta_0\mathbf{H}]$.
`bianisotropic_clausius_mossotti_polarizability(C6, volume)` returns

```math
\boldsymbol{\alpha}_6 = 3V\,(\mathsf{C}_6 - \mathbf{I})\,(\mathsf{C}_6 + 2\mathbf{I})^{-1},
```

rescaled by $\mathrm{diag}(1, 1/\eta_0)\cdots\mathrm{diag}(1, \eta_0)$ so that the
returned polarizability acts on the solver fields $[\mathbf{E}; \mathbf{H}]$ and
returns $[\mathbf{q}; \mathbf{m}]$. Wrap a validated matrix in
`BianisotropicPolarizability3D`, or start from a `BianisotropicMaterial3D`
constitutive tensor (see the
[Material Models API](../api/material-models-3d.md)). The denominator is singular
when $\det(\mathsf{C}_6 + 2\mathbf{I}) \approx 0$.

---

## 8. Worked Example

The following self-contained script reproduces the
`test/test_mom3d.jl` "Voxelized small dielectric sphere polarizability" check: it
carves a small dielectric sphere out of a $7^3$ voxel box, solves the
coupled-dipole system with matrix-free GMRES, and validates both the total
induced dipole and the bistatic RCS against the analytic Rayleigh and exact
dielectric-Mie references in the low-frequency limit. It runs in a few seconds
with `julia --project`.

```julia
using DiffMoM
using LinearAlgebra

# --- Geometry and excitation (deliberately tiny: 7x7x7 grid, low frequency) ---
a = 0.05                       # half-extent of the cube / target sphere radius (m)
lambda = 10.0                  # wavelength (m) -> low frequency, k*a = pi/100 << 1
k0 = 2pi / lambda              # background wavenumber (rad/m)
eps_sphere = 2.5 + 0.0im       # isotropic dielectric

grid = VoxelGrid3D((-a, a), (-a, a), (-a, a), 7, 7, 7)
println("Voxels: ", grid.nvoxels, "  (dx = ", round(grid.dx; digits=5), " m)")
println("k0*a   = ", round(k0 * a; digits=6), "  (Rayleigh regime)")

# Carve a sphere of eps_sphere out of vacuum (eps_r = 1 voxels have alpha = 0)
epsv = ones(ComplexF64, grid.nvoxels)
inside = 0
for j in 1:grid.nvoxels
    if norm(grid.centers[j]) <= a
        epsv[j] = eps_sphere
        global inside += 1
    end
end
println("Inside voxels: ", inside, " / ", grid.nvoxels)

# Plane wave: propagates +z, polarized along x.  E_inc = pol * E0 * exp(-i k.r)
E_inc = planewave_dda_3d(grid, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))

# Matrix-free GMRES solve of  E_i - sum_{j!=i} G_EE(r_i,r_j) alpha_j E_j = E_inc_i
res = solve_dda_3d(grid, k0, epsv, E_inc; solver=:gmres, tol=1e-9, maxiter=200)
println("Solver: ", res.solver, "  GMRES niter = ", res.stats.niter,
        "  solved = ", res.stats.solved)

# --- Check 1: total induced dipole vs analytic Rayleigh polarizability ---
q_total = sum(induced_dipoles_dda_3d(res))           # sum of alpha_j * E_j
alpha_rayleigh = 4pi * a^3 * (eps_sphere - 1) / (eps_sphere + 2)
rel_err_dipole = abs(q_total[1] - alpha_rayleigh) / abs(alpha_rayleigh)
println("\nTotal dipole (x):   ", round(q_total[1]; digits=8))
println("Rayleigh dipole:    ", round(alpha_rayleigh; digits=8))
println("rel err (dipole):   ", round(rel_err_dipole; sigdigits=4))
println("cross-pol leakage:  ",
        round(abs(q_total[2]) / abs(q_total[1]); sigdigits=3), ", ",
        round(abs(q_total[3]) / abs(q_total[1]); sigdigits=3))

# --- Check 2: far-field bistatic RCS vs exact dielectric Mie + Rayleigh ---
rhat = Vec3(0.0, 1.0, 0.0)                            # broadside observation
F_dda = farfield_dda_3d(res, rhat)
sigma_dda = 4pi * real(dot(F_dda, F_dda))
sigma_mie = mie_bistatic_rcs_dielectric(k0, a, Vec3(0.0, 0.0, 1.0),
                                        Vec3(1.0, 0.0, 0.0), rhat, eps_sphere)
sigma_rayleigh = 4pi * k0^4 * a^6 * abs2((eps_sphere - 1) / (eps_sphere + 2))
println("\nsigma_dda      = ", round(sigma_dda; sigdigits=6), " m^2")
println("sigma_mie      = ", round(sigma_mie; sigdigits=6), " m^2")
println("sigma_rayleigh = ", round(sigma_rayleigh; sigdigits=6), " m^2")
println("rel err (DDA vs Mie):       ", round(abs(sigma_dda - sigma_mie) / sigma_mie; sigdigits=4))
println("rel err (Mie vs Rayleigh):  ", round(abs(sigma_mie - sigma_rayleigh) / sigma_rayleigh; sigdigits=4))

# --- Assertions mirroring the test thresholds ---
@assert inside > 0
@assert rel_err_dipole < 0.02            "dipole vs Rayleigh too large"
@assert abs(q_total[2]) / abs(q_total[1]) < 1e-9
@assert abs(q_total[3]) / abs(q_total[1]) < 1e-9
@assert abs(sigma_mie - sigma_rayleigh) / sigma_rayleigh < 1e-3
@assert abs(sigma_dda - sigma_mie) / sigma_mie < 0.06

println("\nOK: voxelized dielectric DDA agrees with the Rayleigh/Mie low-frequency limit.")
```

Running this prints (numbers reproduced from an actual run):

```text
Voxels: 343  (dx = 0.01429 m)
k0*a   = 0.031416  (Rayleigh regime)
Inside voxels: 179 / 343
Solver: gmres  GMRES niter = 14  solved = true

Total dipole (x):   0.00052622 - 0.0im
Rayleigh dipole:    0.0005236 + 0.0im
rel err (dipole):   0.005013
cross-pol leakage:  2.16e-18, 8.35e-19

sigma_dda      = 3.43373e-9 m^2
sigma_mie      = 3.40067e-9 m^2
sigma_rayleigh = 3.40022e-9 m^2
rel err (DDA vs Mie):       0.009724
rel err (Mie vs Rayleigh):  0.0001315

OK: voxelized dielectric DDA agrees with the Rayleigh/Mie low-frequency limit.
```

The staircased $7^3$ sphere reproduces the Rayleigh dipole to $\sim 0.5\%$ and
the exact dielectric-Mie RCS to $\sim 1\%$, with essentially zero cross-polarized
leakage -- exactly what the volume-equivalence derivation predicts for an
isotropic, low-frequency target. Refining the grid drives both errors toward
zero.

### 8.1 Switching to the FFT Operator

For larger grids, swap the matrix-free operator for the FFT-accelerated one and
drive GMRES with `Krylov` directly. The matvec matches `dda_operator_3d` to
machine precision while scaling as $O(N\log N)$:

```julia
using DiffMoM, Krylov, LinearAlgebra

A_fft = fft_dda_operator_3d(grid, k0, epsv)
rhs = reduce(vcat, E_inc)                       # flatten Vector{CVec3} -> 3N vector
x, stats = Krylov.gmres(A_fft, rhs; rtol=1e-9, atol=0.0, itmax=200)
@assert stats.solved
```

---

## 9. Validation and When to Use Each Operator

### 9.1 Validation Evidence in the Repo

The subsystem is exercised by an extensive test suite; all numbers below are
asserted in the repository (not estimated here):

- **Voxelized dielectric sphere** (`test/test_mom3d.jl`): the Worked Example case. Asserts dipole-vs-Rayleigh relative error $< 0.02$, cross-pol leakage $< 10^{-10}$, Mie-vs-Rayleigh $< 10^{-3}$, and DDA-vs-Mie $< 0.06$.
- **Free-space limit** (`test/test_mom3d.jl`): with $\varepsilon_r = 1$, $\alpha = 0$, the total field equals the incident field to $< 10^{-13}$ and the scattered field is $\approx 0$.
- **Reciprocal dyadic symmetry** (`test/test_mom3d.jl`): off-diagonal `3 x 3` blocks satisfy $\mathbf{G}_{12} = \mathbf{G}_{21}^{\!\top}$ to $< 10^{-13}$.
- **Single-voxel Rayleigh dipole far field** and **anisotropic tensor polarizability** (`test/test_mom3d.jl`): closed-form checks to $\sim 10^{-13}$ / $10^{-16}$.
- **Matrix-free equivalence** (`test/test_mom3d.jl`): the `DDAOperator3D` matvec matches dense `assemble_dda_3d` to $< 10^{-13}$, storage $<$ dense$/20$, matvec allocates $< 1024$ bytes; matrix-free GMRES agrees with dense direct to $< 10^{-10}$.
- **Coupled EM** (`test/test_mom3d_em.jl`): free-space magnetodielectric limit, electric-only reduction to $< 10^{-13}$, single-voxel magnetic response, explicit bianisotropic `6 x 6` closure, dense-vs-matrix-free equivalence, and the radiation condition $\mathbf{F}_E = -\eta_0(\hat{\mathbf{n}}\times\mathbf{F}_H)$ to $< 10^{-10}$.
- **FFT operators** (`test/test_mom3d_fft.jl`): `fft_dda_operator_3d` / `fft_em_dda_operator_3d` match the dense/direct matvec to $< 10^{-12}$ for scalar, tensor, and the EM `6 x 6` case.
- **Adjoint** (`test/test_mom3d_adjoint.jl`): `gradient_epsr_dda_3d` matches central finite differences of $J = \Re(\mathbf{E}^{\dagger}\,\mathrm{diag}(w)\,\mathbf{E})$ to `rtol=2e-5`; the gradient is real; GMRES adjoint matches the direct adjoint to $< 10^{-6}$.
- **Full Mie sweep** (`validation/mie/validate_dielectric_mie_dda.jl`): a dielectric-sphere DDA-vs-exact-Mie bistatic RCS sweep (default $11^3$ grid, $ka = 0.6$, $\varepsilon_r = 2.5 - 0.02i$) over $\theta$ in the $\phi = 0$ and $\phi = 90$ cuts; passes if MAE $<$ 0.4 dB, RMSE $<$ 0.7 dB, max $|\Delta| <$ 3.0 dB on both cuts. The agreement improves qualitatively under refinement.

### 9.2 Choosing an Operator

| Operator | Cost / matvec | Storage | Use when |
|----------|---------------|---------|----------|
| Dense (`assemble_dda_3d`, `solver=:direct`) | $O(N^2)$ apply, $O(N^3)$ factor | $O(N^2)$ | tiny grids; you also want the LU for a cheap adjoint |
| Matrix-free (`dda_operator_3d`, `solver=:gmres`) | $O(N^2)$ | $O(N)$, near-zero alloc | any non-tiny grid that still fits $O(N^2)$ matvecs |
| FFT (`fft_dda_operator_3d` + Krylov) | $O(N\log N)$ | $O(N)$ + padded spectrum | large uniform grids |

The default `solver=:direct` factorizes a dense matrix -- pass `solver=:gmres`
for anything beyond a few hundred voxels, and switch to the FFT operator as the
voxel count grows. The matrix-free `mul!` is intentionally *not* threaded to
hold the tight allocation budget asserted by the tests.

### 9.3 Pitfalls

- **Coincident points.** `electric_dipole_dyadic_3d` and `scattered_field_dda_3d` are singular at $\mathbf{r} = \mathbf{r}'$; keep observation points off the voxel centers. The self term lives only in $\alpha$.
- **Transverse polarization.** `planewave_dda_3d` requires $|\hat{\mathbf{k}}\cdot\hat{\mathbf{p}}| \le 10^{-10}$; a non-transverse polarization errors.
- **Clausius-Mossotti singularity.** $\varepsilon_r$ (or $\mu_r$, or $\mathsf{C}_6$) near the $-2$ resonance makes the polarizability singular and raises an error.
- **Grid validity.** `VoxelGrid3D` requires $n_x, n_y, n_z \ge 1$ and strictly increasing ranges. The operator constructors require $k_0 > 0$.
- **Finite-voxel accuracy.** The $3V$ Clausius-Mossotti form matches the smooth-sphere Rayleigh reference only in the low-frequency limit and improves under refinement -- it is not exact at a finite voxel count.

---

## 10. Adjoint Sensitivity for Permittivity Design

For gradient-based design of a real per-voxel permittivity, the DDA path
provides an adjoint sensitivity that mirrors the surface-EFIE adjoint workflow.
Given an objective $J = \Re(\mathbf{E}^{\dagger}\,\mathbf{Q}\,\mathbf{E})$:

1. Solve the forward system for $\mathbf{E}$.
2. Solve the adjoint system $A^{\dagger}\boldsymbol{\lambda} = \mathbf{Q}\mathbf{E}$ with `solve_dda_adjoint_3d` (the `:direct` mode reuses the stored LU from the forward solve).
3. Form the real gradient with `gradient_epsr_dda_3d`:

```math
\frac{\partial J}{\partial \varepsilon_{r,j}}
   = 2\,\Re\!\left(
        \frac{\partial \alpha_j}{\partial \varepsilon_{r,j}}
        \sum_{i \neq j} \boldsymbol{\lambda}_i \cdot
        \mathbf{G}_{ij}\,\mathbf{E}_j
      \right),
\qquad
\frac{\partial \alpha}{\partial \varepsilon_r} = \frac{9V}{(\varepsilon_r + 2)^2}.
```

```julia
res = solve_dda_3d(grid, k0, epsv, E_inc)          # :direct keeps the LU
E = reduce(vcat, res.E_total)
lambda = solve_dda_adjoint_3d(res, weights .* E)   # grad_E_flat = Q * E
grad = gradient_epsr_dda_3d(res, lambda)           # dJ/d eps_r per voxel (real)
```

The adjoint supports **only the uncorrected** Clausius-Mossotti $\alpha$ (it
errors if `radiative_correction=true`) and differentiates a single real scalar
$\varepsilon_r$ per voxel, not tensors.

---

## 11. Code Mapping

| Concept | Exported function / type | Source file |
|---------|--------------------------|-------------|
| Voxel grid | `VoxelGrid3D`, `make_voxel_grid_3d` | `src/mom3d/Types3D.jl` |
| Polarizability | `clausius_mossotti_polarizability`, `dda_polarizabilities` | `src/mom3d/DDA3D.jl` |
| Dipole dyadic | `electric_dipole_dyadic_3d` | `src/mom3d/DDA3D.jl` |
| Electric operators | `dda_operator_3d`, `assemble_dda_3d`, `DDAOperator3D` | `src/mom3d/DDA3D.jl`, `src/mom3d/Types3D.jl` |
| Excitation | `planewave_dda_3d` | `src/mom3d/DDA3D.jl` |
| Solve | `solve_dda_3d` | `src/mom3d/DDA3D.jl` |
| Observables | `induced_dipoles_dda_3d`, `scattered_field_dda_3d`, `farfield_dda_3d` | `src/mom3d/DDA3D.jl` |
| EM / bianisotropic | `magnetic_clausius_mossotti_polarizability`, `bianisotropic_clausius_mossotti_polarizability`, `em_dda_operator_3d`, `solve_em_dda_3d`, `planewave_em_dda_3d`, `farfield_em_dda_3d`, `BianisotropicPolarizability3D` | `src/mom3d/EMDDA3D.jl` |
| FFT acceleration | `fft_dda_operator_3d`, `fft_em_dda_operator_3d` | `src/mom3d/FFTDDA3D.jl` |
| Adjoint / gradient | `solve_dda_adjoint_3d`, `gradient_epsr_dda_3d` | `src/mom3d/Adjoint3D.jl` |
| Reference oracle | `mie_bistatic_rcs_dielectric` | `src/postprocessing/Mie.jl` |

Full signatures and field tables are in the
[3D Volume DDA API page](../api/dda-volume-3d.md); for constitutive material
inputs (magnetic and bianisotropic tensors) see the
[Material Models API](../api/material-models-3d.md).

---

## 12. Exercises

### 12.1 Conceptual

1. **Sign bookkeeping.** Starting from $e^{+i\omega t}$, show why the outgoing scalar Green function carries $e^{-ikR}$, the plane wave carries $e^{-i\mathbf{k}\cdot\mathbf{r}}$, and the far-field dipole sum carries the conjugate phase $e^{+ik\hat{\mathbf{r}}\cdot\mathbf{r}_j}$.
2. **Why no self term?** Explain why $\mathbf{G}_{EE}$ is excluded for $i = j$ and how the polarizability $\alpha$ stands in for the self interaction. What goes wrong if you naively include the singular diagonal block?
3. **$3V$ vs $4\pi a^3$.** Reconcile the volume coefficient $3V$ in the normalized Clausius-Mossotti polarizability with the $4\pi a^3$ in the single-sphere Rayleigh reference. In what limit do they agree?

### 12.2 Numerical

4. **Grid refinement.** Re-run the Worked Example at $5^3$, $7^3$, $9^3$, and $11^3$ and tabulate `rel_err_dipole` and the DDA-vs-Mie RCS error. Confirm both decrease under refinement.
5. **Operator equivalence.** For the Worked Example grid, build `dda_operator_3d` and `fft_dda_operator_3d`, apply both to a random complex vector, and verify `norm(y_fft - y_mf) / norm(y_mf)` is at machine precision.
6. **Lossy dielectric.** Set `eps_sphere = 2.5 - 0.1im` and compare the broadside RCS with `mie_bistatic_rcs_dielectric` at the same complex $\varepsilon_r$.

### 12.3 Advanced

7. **Magnetic scatterer.** Build a magnetodielectric sphere ($\varepsilon_r = 2.3$, $\mu_r = 1.4$) with `planewave_em_dda_3d` and `solve_em_dda_3d`, then verify the deep far-zone radiation condition $\mathbf{F}_E = -\eta_0(\hat{\mathbf{n}}\times\mathbf{F}_H)$ from `farfield_em_dda_3d`.
8. **Permittivity gradient check.** Pick a small grid, define $J = \Re(\mathbf{E}^{\dagger}\mathbf{Q}\mathbf{E})$ with a diagonal weight, and verify `gradient_epsr_dda_3d` against a central finite difference of $J$ with respect to one voxel's $\varepsilon_r$.

---

## 13. Chapter Checklist

Before moving on, make sure you can:

- [ ] State the $e^{+i\omega t}$ / $e^{-ikR}$ convention and the three signs it fixes.
- [ ] Write the coupled-dipole system $\mathbf{E}_i - \sum_{j\neq i}\mathbf{G}_{ij}\alpha_j\mathbf{E}_j = \mathbf{E}^{\text{inc}}_i$ and its block form $A_{ij} = \delta_{ij}\mathbf{I} - \mathbf{G}_{ij}\alpha_j$.
- [ ] Reproduce the electric dipole dyadic and explain its radiation vs near terms.
- [ ] Compute the normalized Clausius-Mossotti polarizability and recognize the $\varepsilon_r = -2$ singularity.
- [ ] Excite, solve (`:direct` / `:gmres` / FFT), and post-process induced dipoles and far fields.
- [ ] Extend to the coupled `6 x 6` EM DDA and the bianisotropic mapping.
- [ ] Compute a permittivity design gradient with the DDA adjoint.

---

## 14. Further Reading

1. **Discrete-dipole approximation:**
   - Draine, B. T. & Flatau, P. J. (1994). "Discrete-dipole approximation for scattering calculations." *J. Opt. Soc. Am. A*, 11(4), 1491-1499. The standard reference for DDA, the Clausius-Mossotti dipole rule, and the radiative correction.
   - Yurkin, M. A. & Hoekstra, A. G. (2007). "The discrete dipole approximation: An overview and recent developments." *J. Quant. Spectrosc. Radiat. Transf.*, 106, 558-589.

2. **Volume integral equations:**
   - Markkanen, J. & Yla-Oijala, P. (2016). "Numerical comparison of spectral properties of volume-integral-equation formulations." *J. Quant. Spectrosc. Radiat. Transf.*, 178, 269-275.

3. **FFT acceleration on uniform grids:**
   - Goodman, J. J., Draine, B. T. & Flatau, P. J. (1991). "Application of fast-Fourier-transform techniques to the discrete-dipole approximation." *Opt. Lett.*, 16(15), 1198-1200.

---

*Related: the [3D Volume DDA API page](../api/dda-volume-3d.md) gives full
signatures; [Material Models](../api/material-models-3d.md) covers the
constitutive tensors; [Physical Optics](../methods/03-physical-optics.md) and
[MLFMA](../methods/06-mlfma.md) cover the surface-EFIE high-frequency and
large-problem methods.*
