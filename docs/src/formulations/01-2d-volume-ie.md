# 2D Volume Integral Equation (TM Polarization)

## Purpose

Most of `DiffMoM.jl` solves *surface* integral equations on perfect conductors and homogeneous dielectrics. But many problems involve an **inhomogeneous** dielectric region -- a graded-index lens, a photonic structure, a permittivity profile we want to design -- where the material varies continuously through the volume. For these, the natural formulation is a **volume integral equation (VIE)**: the unknown lives throughout the dielectric domain, and the contrast between the material and free space acts as a distributed source that radiates the scattered field.

This chapter develops the 2D TM (transverse-magnetic, $E_z$-only) contrast-source VIE implemented in `src/mom2d/`. We derive the Lippmann-Schwinger equation from the scalar Helmholtz equation, discretize it with a pulse basis and point matching on a uniform Cartesian grid, build the system matrix $Z = I - k_0^2\,D\,\mathrm{diag}(\chi)$, and recover the scattered field by radiating the contrast currents. We also cover the contrast Jacobian $\partial E_\text{scat}/\partial\chi$ (the gradient backbone for differentiable inverse design) and validate everything against the analytical 2D Mie series for a circular cylinder. This is the simplest fully worked VIE in the package and a good on-ramp to volumetric and inverse-design methods.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive the TM contrast-source (Lippmann-Schwinger) VIE from the scalar Helmholtz equation, identifying $\chi = \varepsilon_r - 1$ as the equivalent volume source.
2. State the 2D scalar free-space Green's function and explain why the $\exp(+i\omega t)$ convention forces the Hankel function of the *second* kind.
3. Discretize the VIE with pulse basis + point matching, and assemble $Z = I - k_0^2\,D\,\mathrm{diag}(\chi)$.
4. Explain why the diagonal of the Green's matrix $D$ needs a special analytical self-cell integral while the off-diagonal uses a midpoint rule.
5. Build plane-wave and line-source excitations and recover the scattered field by radiating the contrast currents.
6. Compute the contrast Jacobian via implicit differentiation and verify it against finite differences.
7. Validate the solver against the 2D Mie series and recognize the accuracy limits imposed by staircasing and the midpoint quadrature.

---

## 1. From the Helmholtz Equation to a Volume Integral Equation

### 1.1 The TM Scalar Problem

For TM polarization in two dimensions, the only nonzero electric-field component is $E_z(x,y)$, and Maxwell's equations collapse to a scalar Helmholtz equation. In a region with relative permittivity $\varepsilon_r(\mathbf{r})$ (nonmagnetic, $\mu_r = 1$), the total field satisfies

```math
\left(\nabla^2 + k_0^2\,\varepsilon_r(\mathbf{r})\right) E_z(\mathbf{r}) = 0,
\qquad k_0 = \frac{\omega}{c} = \frac{2\pi}{\lambda},
```

where $k_0$ is the free-space wavenumber. Splitting the wavenumber into a background part plus a perturbation,

```math
k_0^2\,\varepsilon_r(\mathbf{r}) = k_0^2 + k_0^2\bigl(\varepsilon_r(\mathbf{r}) - 1\bigr) = k_0^2 + k_0^2\,\chi(\mathbf{r}),
```

we define the **dielectric contrast**

```math
\chi(\mathbf{r}) = \varepsilon_r(\mathbf{r}) - 1.
```

The contrast vanishes in free space ($\varepsilon_r = 1 \Rightarrow \chi = 0$) and is nonzero only inside the scatterer. Rearranging the Helmholtz equation isolates the background operator on the left and treats the contrast term as a source:

```math
\left(\nabla^2 + k_0^2\right) E_z(\mathbf{r}) = -k_0^2\,\chi(\mathbf{r})\,E_z(\mathbf{r}).
```

The right-hand side is an **equivalent volume current** that exists only where $\chi \neq 0$ and is proportional to the (unknown) total field. This is the contrast-source picture: the dielectric reradiates as a distributed source driven by the field inside it.

### 1.2 The Lippmann-Schwinger Equation

Inverting the background Helmholtz operator with the free-space Green's function $G_\text{2D}$ (Section 2) and adding the homogeneous solution (the incident field) gives the **Lippmann-Schwinger volume integral equation**:

```math
E_z(\mathbf{r}) = E_z^\text{inc}(\mathbf{r})
+ k_0^2 \int_D \chi(\mathbf{r}')\, G_\text{2D}(\mathbf{r}, \mathbf{r}')\, E_z(\mathbf{r}')\, dA'.
```

This is the governing equation of the subsystem (`src/mom2d/Assembly2D.jl` header; `docs/src/api/vie-2d.md`). The integration domain $D$ is any region enclosing the support of $\chi$. The unknown is the **total** field $E_z$ *inside* $D$; once known, the same integral evaluated at any exterior point yields the scattered field (Section 6).

Note the structure: $E_z$ appears both outside the integral (the total field we solve for) and inside it (the contrast source). This is a Fredholm integral equation of the second kind -- exactly the form that produces a well-conditioned $I - (\text{compact operator})$ system after discretization.

---

## 2. The 2D Scalar Green's Function and Time Convention

### 2.1 Definition

The 2D scalar free-space Green's function is the outgoing solution of

```math
\left(\nabla^2 + k^2\right) G_\text{2D}(\mathbf{r}, \mathbf{r}') = -\delta(\mathbf{r} - \mathbf{r}').
```

In `DiffMoM.jl` it is implemented (`src/mom2d/Greens2D.jl`) as

```math
G_\text{2D}(\mathbf{r}, \mathbf{r}') = -\frac{i}{4}\, H_0^{(2)}\!\bigl(k\,|\mathbf{r} - \mathbf{r}'|\bigr),
```

where $H_0^{(2)}$ is the zeroth-order Hankel function of the **second** kind.

### 2.2 Why the Second Kind?

The package uses the $\exp(+i\omega t)$ time convention throughout (stated in every source header, e.g. `Types2D.jl`, `Greens2D.jl`, `Excitation2D.jl`, `Mie2D.jl`). With $\exp(+i\omega t)$, an outgoing cylindrical wave behaves as

```math
H_0^{(2)}(kr) \sim \sqrt{\frac{2}{\pi k r}}\, e^{-i(kr - \pi/4)} \quad (kr \to \infty),
```

so the full time-space dependence $e^{+i\omega t} e^{-ikr}$ describes a wave traveling **outward** (constant phase $\omega t - kr$ moves to larger $r$). The Hankel function of the first kind $H_0^{(1)}$ would describe an incoming wave under this convention and would violate the radiation condition. Consequently, **every** radiating quantity in this subsystem uses `besselh(n, 2, ...)`: the Green's function ($-i/4\,H_0^{(2)}$), the self-cell term ($H_1^{(2)}$, Section 4), and the Mie series ($H_n^{(2)}$, Section 7). The matching spatial phase for a plane wave is $\exp(-ik_0\,\hat{\mathbf{k}}\cdot\mathbf{r})$ (Section 5).

### 2.3 The Coincident-Point Guard

The Green's function is singular as $\mathbf{r}\to\mathbf{r}'$ (it diverges logarithmically). The implementation returns **exactly zero** when the separation falls below $10^{-30}$ rather than `Inf`:

```julia
function greens_2d(r::Vec2, rp::Vec2, k::Float64)
    R = sqrt(dot(r - rp, r - rp))
    R < 1e-30 && return zero(ComplexF64)
    return (-im / 4) * besselh(0, 2, k * R)
end
```

This is a deliberate sentinel: the true self-interaction (the diagonal of the system) is *not* the value of $G_\text{2D}$ at zero separation -- it is the integral of $G_\text{2D}$ over the cell, supplied separately by `self_cell_integral_2d` (Section 4). Never use `greens_2d` for a self term.

---

## 3. Discretization: Pulse Basis and Point Matching

### 3.1 The Mesh

The domain is tiled by a uniform Cartesian grid (`Mesh2D`, `src/mom2d/Types2D.jl`). Each cell carries a single constant material value and a single constant field value -- this is the **pulse (piecewise-constant) basis**. The constructor `Mesh2D(x_range, y_range, nx, ny)` places cell centers at the midpoint of each cell,

```math
\mathbf{r}_{(ix,iy)} = \Bigl(x_0 + (ix - \tfrac{1}{2})\,dx,\;\; y_0 + (iy - \tfrac{1}{2})\,dy\Bigr),
```

stored row-major ($ix$ fastest, then $iy$), with uniform cell area $A = dx\,dy$ and total cell count `ncells = nx*ny`. The constructor asserts `nx >= 1`, `ny >= 1`, and strictly increasing ranges.

### 3.2 Expanding the Unknown

Write the unknown total field as a sum of pulses $\Pi_n(\mathbf{r})$ (unity on cell $n$, zero elsewhere) with constant coefficients $E_n$:

```math
E_z(\mathbf{r}) \approx \sum_{n=1}^{N} E_n\,\Pi_n(\mathbf{r}), \qquad N = \text{ncells}.
```

The contrast is likewise piecewise constant, $\chi(\mathbf{r}) \approx \chi_n$ on cell $n$. Substituting into the Lippmann-Schwinger equation:

```math
\sum_n E_n\,\Pi_n(\mathbf{r}) = E_z^\text{inc}(\mathbf{r})
+ k_0^2 \sum_n \chi_n\, E_n \int_{\text{cell}_n} G_\text{2D}(\mathbf{r}, \mathbf{r}')\, dA'.
```

### 3.3 Point Matching (Collocation)

We enforce this equation at the $N$ cell centers $\mathbf{r}_m$ (point matching / collocation). Evaluating the pulse sum at $\mathbf{r}_m$ picks out $E_m$, giving one equation per cell:

```math
E_m = E_m^\text{inc} + k_0^2 \sum_n \chi_n\, E_n\, D_{mn},
\qquad
D_{mn} \equiv \int_{\text{cell}_n} G_\text{2D}(\mathbf{r}_m, \mathbf{r}')\, dA'.
```

Here $D_{mn}$ is the **Green's integral matrix**: the field at center $m$ due to a uniform unit source over cell $n$. Collecting the equations into matrix form:

```math
\bigl(I - k_0^2\,D\,\mathrm{diag}(\chi)\bigr)\,\mathbf{E} = \mathbf{E}^\text{inc}.
```

This defines the **system matrix**

```math
\boxed{\,Z = I - k_0^2\,D\,\mathrm{diag}(\chi)\,}, \qquad Z_{mn} = \delta_{mn} - k_0^2\,\chi_n\,D_{mn},
```

assembled exactly this way in `assemble_vie_2d` (`src/mom2d/Assembly2D.jl`): the loop fills $-k_0^2\,\chi_n D_{mn}$ then adds $1$ on the diagonal. Note the ordering: $\chi_n$ multiplies column $n$ of $D$, so the diagonal factor sits on the **right** -- $D\,\mathrm{diag}(\chi)$ scales the columns of $D$, giving $(D\,\mathrm{diag}(\chi))_{mn} = D_{mn}\,\chi_n$. Left-multiplying instead, $(\mathrm{diag}(\chi)\,D)_{mn} = \chi_m\,D_{mn}$, scales rows and does **not** match the code. Since $\mathrm{diag}(\chi)\,D \neq D\,\mathrm{diag}(\chi)$ in general, the right-multiplied form is the correct one. When $\chi = 0$ (free space), $Z$ reduces to the identity, so $\mathbf{E}_\text{total} = \mathbf{E}^\text{inc}$ -- a sanity check the test suite asserts to `atol=1e-14`.

`solve_vie_2d` assembles $Z$, factorizes it once with an LU decomposition, solves $Z\,\mathbf{E}_\text{total} = \mathbf{E}^\text{inc}$, and bundles everything (including the cached `Z_LU`) into a `VIEResult2D` for downstream reuse.

---

## 4. The Green's Integral Matrix $D$

The quality of the whole solve rests on how accurately $D_{mn}$ is computed. The implementation (`assemble_D_matrix`, `src/mom2d/Greens2D.jl`) splits into two cases.

### 4.1 Off-Diagonal: Midpoint Rule

For $m \neq n$ the source and observation cells are distinct, $G_\text{2D}$ is smooth over cell $n$, and a single-point midpoint (one-point quadrature) rule is used:

```math
D_{mn} \approx G_\text{2D}(\mathbf{r}_m, \mathbf{r}_n)\, A, \qquad m \neq n.
```

This collapses the integral to the Green's value at the cell center times the cell area. It is accurate only when cells are small relative to the wavelength; accuracy degrades for coarse or electrically large cells. Because $G_\text{2D}$ depends only on $|\mathbf{r}_m - \mathbf{r}_n|$ and is symmetric, $D$ is symmetric ($D = D^\top$) -- the discrete statement of reciprocity, asserted in the tests to `atol=1e-13`.

### 4.2 Diagonal: Analytical Self-Cell Integral

For $m = n$ the midpoint rule fails -- it would evaluate $G_\text{2D}$ at zero separation, where it is singular (and where `greens_2d` returns the zero sentinel). Instead, the self term integrates $G_\text{2D}$ analytically over an **area-equivalent disk** that replaces the square cell. The equivalent radius $a_\text{eq}$ is chosen so the disk has the same area as the cell:

```math
\pi\, a_\text{eq}^2 = A \quad\Longrightarrow\quad a_\text{eq} = \sqrt{A/\pi},
```

implemented as `equivalent_radius(mesh)`. The integral over the disk has a closed form (`self_cell_integral_2d`):

```math
\int_{|\mathbf{r}'|\le a_\text{eq}} G_\text{2D}(\mathbf{0}, \mathbf{r}')\, dA'
= -\frac{i\pi}{2k^2}\left[\,k\,a_\text{eq}\, H_1^{(2)}(k\,a_\text{eq}) - \frac{2i}{\pi}\,\right].
```

This follows from the identity $\frac{d}{du}\!\left[u\,H_1^{(2)}(u)\right] = u\,H_0^{(2)}(u)$, which lets the radial integral $\int_0^{a_\text{eq}} H_0^{(2)}(k\rho)\,\rho\,d\rho$ be done in closed form (the $-2i/\pi$ term comes from the small-argument behavior of $H_1^{(2)}$). The result is finite with nonzero real and imaginary parts, and the function asserts `k > 0` and `a_eq > 0`.

The diagonal of $D$ is the single value $D_\text{self}$, returned by `self_cell_integral_2d(k, a_eq)` (the same for every cell on a uniform grid); the off-diagonals are midpoint values. This is the only singular integration in the formulation.

---

## 5. Excitation

### 5.1 Plane Wave

A TM plane wave propagating along $\hat{\mathbf{k}} = (\cos\phi_\text{inc}, \sin\phi_\text{inc})$ is sampled at the cell centers by `planewave_2d` (`src/mom2d/Excitation2D.jl`):

```math
E_z^\text{inc}(\mathbf{r}) = E_0\, e^{-i k_0\,\hat{\mathbf{k}}\cdot\mathbf{r}}.
```

The negative spatial phase matches the $\exp(+i\omega t)$ convention (Section 2). The default amplitude is $E_0 = 1$, and `phi_inc = 0` means propagation along $+x$, so $E_z^\text{inc} = e^{-ik_0 x}$ (unit amplitude, asserted in the tests to `atol=1e-14`).

### 5.2 Line Source

A unit-amplitude 2D line source at $\mathbf{r}_\text{src}$ produces a cylindrical incident field identical to the Green's function (`linesource_2d`):

```math
E_z^\text{inc}(\mathbf{r}) = -\frac{i}{4}\, H_0^{(2)}\!\bigl(k_0\,|\mathbf{r} - \mathbf{r}_\text{src}|\bigr) = G_\text{2D}(\mathbf{r}, \mathbf{r}_\text{src}).
```

The source must lie **outside** the scattering domain (the field is singular at $\mathbf{r}_\text{src}$).

---

## 6. Recovering the Scattered Field

Once the internal total field $\mathbf{E}_\text{total}$ is known, each cell becomes a known contrast source. The scattered field at any **exterior** observation point $\mathbf{r}_\text{obs}$ is the discretized contrast-source integral (`scattered_field_2d`, `src/mom2d/Scatter2D.jl`):

```math
E_z^\text{scat}(\mathbf{r}_\text{obs}) = k_0^2 \sum_{n=1}^{N} \chi_n\, E_n\, G_\text{2D}(\mathbf{r}_\text{obs}, \mathbf{r}_n)\, A,
\qquad E_n = (\mathbf{E}_\text{total})_n,\;\; A = \text{cell\_area}.
```

The Green's values $G_\text{obs}[m,n] = G_\text{2D}(\mathbf{r}_{\text{obs},m}, \mathbf{r}_n)$ are assembled by `green_obs_matrix`. Observation points must lie outside the domain; if $\mathbf{r}_\text{obs}$ coincides with a cell center the midpoint evaluation is singular.

> **Note.** The VIE path models lossless real-$\varepsilon_r$ dielectrics only: `chi` is a `Vector{Float64}` (real contrast). There is no lossy or PEC support in the VIE itself -- PEC appears only in the Mie oracle via `pec=true` (Section 7).

---

## 7. The Contrast Jacobian (Differentiable Design)

For inverse design we want the sensitivity of the measured scattered field to the per-cell contrast, $J_{mp} = \partial E_z^\text{scat}(\mathbf{r}_{\text{obs},m})/\partial \chi_p$. Differentiating finite differences over a fresh solve per cell would cost $N$ extra factorizations; instead the package uses **implicit differentiation** of the system (`jacobian_scattered_field_2d`).

Differentiating $Z\,\mathbf{E} = \mathbf{E}^\text{inc}$ with respect to $\chi_p$ (the incident field does not depend on $\chi$) gives

```math
\frac{\partial Z}{\partial \chi_p}\,\mathbf{E} + Z\,\frac{\partial \mathbf{E}}{\partial \chi_p} = \mathbf{0}.
```

Since $Z = I - k_0^2\,D\,\mathrm{diag}(\chi)$, only column $p$ of $D\,\mathrm{diag}(\chi)$ (which equals $\chi_p\,D[:,p]$) depends on $\chi_p$, so $\partial Z/\partial\chi_p$ has the single nonzero column $-k_0^2 D[:,p]$. Therefore

```math
\frac{\partial \mathbf{E}}{\partial \chi_p} = k_0^2\, E_p\, Z^{-1} D[:,p].
```

The routine precomputes $S = Z^{-1} D$ once (reusing the cached `Z_LU`), forms $W = I + k_0^2\,\mathrm{diag}(\chi)\,S$, and assembles the Jacobian by chaining the field sensitivity through the scattered-field map:

```math
J = k_0^2\, A\; G_\text{obs}\; W\; \mathrm{diag}(\mathbf{E}_\text{total}), \qquad J \in \mathbb{C}^{M \times N}.
```

This reuses the existing factorization, so the Jacobian costs one block solve ($Z^{-1}D$) plus dense products -- no re-factorization. **Correctness depends on `Z_LU` matching the `chi`/`mesh`/`k0` stored in the `VIEResult2D`; do not mutate those fields after the solve.** The test suite validates $J$ column-by-column against finite differences ($\delta = 10^{-7}$) on interior, exterior, and random cells, asserting relative error below $10^{-4}$ (finite-difference-limited).

---

## 8. Validation Oracle: the 2D Mie Series

The independent reference is the analytical 2D Mie series for a homogeneous circular cylinder (`src/mom2d/Mie2D.jl`). For TM incidence the scattered field is a cylindrical-harmonic expansion:

```math
E_z^\text{scat}(\rho, \phi) = E_0 \sum_{n=-N}^{N} (-i)^n\, c_n\, H_n^{(2)}(k_0 \rho)\, e^{\,i n (\phi - \phi_\text{inc})}.
```

The coefficients $c_n$ enforce the boundary conditions at $\rho = a$:

- **PEC cylinder** ($E_z = 0$ on the surface): $c_n = -J_n(k_0 a)/H_n^{(2)}(k_0 a)$.
- **Dielectric cylinder**: continuity of the field and its radial derivative at $\rho = a$, using the interior wavenumber $k_1 = k_0\sqrt{\varepsilon_r}$. Bessel/Hankel derivatives use the recurrence $f_n'(x) = f_{n-1}(x) - (n/x) f_n(x)$.

The truncation order is chosen automatically as $N = \max\!\bigl(10,\, \lceil k_0 a + 4 (k_0 a)^{1/3} + 2 \rceil\bigr)$. The total field outside the cylinder ($\rho > a$) is the incident plane wave plus the scattered series, using the *same* incident convention $e^{-ik_0\,\hat{\mathbf{k}}\cdot\mathbf{r}}$ as `planewave_2d`, so VIE and Mie results are directly comparable.

The Mie tests confirm: the PEC surface field is nullified to $\sim10^{-6}$ (boundary condition satisfied to truncation accuracy), and dielectric coefficients are symmetric $c_{-n} \approx c_n$ for $\phi_\text{inc} = 0$.

---

## Worked Example

The following self-contained script solves the VIE for a small dielectric cylinder, compares the scattered field to the Mie series at two grid resolutions, and runs the internal-consistency checks (free-space identity, $D$ symmetry, self-cell integral, Jacobian vs. finite differences). It runs in well under a second.

```julia
# Worked example: 2D Volume Integral Equation (TM polarization).
# Build a dielectric cylinder approximated on a Cartesian grid, solve the
# contrast-source VIE under plane-wave incidence, and validate the scattered
# field against the analytical 2D Mie series. Convention: exp(+iωt), H0^(2).

using DiffMoM
using LinearAlgebra

# --- Problem setup (small; runs in well under a second) ---
freq   = 1e9
c0     = 3e8
lambda = c0 / freq
k0     = 2π / lambda            # free-space wavenumber (rad/m)

a       = 0.1 * lambda          # cylinder radius (electrically small, k0*a ≈ 0.63)
eps_r   = 4.0                   # relative permittivity
chi_val = eps_r - 1.0           # dielectric contrast χ = εr - 1

# Observation ring outside the scatterer (ρ = 3a > a)
r_obs = [Vec2(3a * cos(phi), 3a * sin(phi))
         for phi in range(0, 2π, length=37)[1:36]]

# Analytical Mie reference (the validation oracle)
E_scat_mie = mie_scattered_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0)

# --- VIE solve on a tiny grid bounding the cylinder ---
# Build a circular dielectric cylinder by setting χ inside ρ ≤ a (staircased).
function run_vie(n)
    mesh = Mesh2D((-a, a), (-a, a), n, n)
    chi  = zeros(mesh.ncells)
    for i in 1:mesh.ncells
        r = sqrt(mesh.centers[i][1]^2 + mesh.centers[i][2]^2)
        r <= a && (chi[i] = chi_val)
    end
    E_inc = planewave_2d(mesh, k0, 0.0)          # E_z^inc = exp(-i k0 x)
    vr    = solve_vie_2d(mesh, k0, chi, E_inc)   # Z E = E_inc, Z = I - k0² D diag(χ)
    E_scat_mom = scattered_field_2d(vr, r_obs)   # E_scat = k0² Σ χ_n E_n G A_n
    return mesh, vr, E_scat_mom
end

println("== 2D VIE (TM) vs analytical Mie series ==")
println("k0 = ", round(k0, sigdigits=6), " rad/m,  a = ", round(a, sigdigits=4),
        " m,  k0*a = ", round(k0 * a, sigdigits=4), ",  eps_r = ", eps_r)

for n in (8, 16)
    mesh, vr, E_scat_mom = run_vie(n)
    rel_err = norm(E_scat_mom - E_scat_mie) / norm(E_scat_mie)
    println("grid $(n)x$(n)  ncells=$(mesh.ncells)  rel-err vs Mie = ",
            round(rel_err, sigdigits=4))
end

# --- Sanity checks on the smallest grid ---
mesh, vr, E_scat_mom = run_vie(8)

# Free-space consistency: χ = 0  ⟹  Z = I  ⟹  E_total = E_inc
E_inc_fs = planewave_2d(mesh, k0, 0.0)
vr_fs    = solve_vie_2d(mesh, k0, zeros(mesh.ncells), E_inc_fs)
@assert vr_fs.E_total ≈ E_inc_fs atol=1e-12
println("free-space check (χ=0 ⟹ E_total=E_inc): OK")

# D matrix symmetry (reciprocity): D = Dᵀ
D = DiffMoM.assemble_D_matrix(mesh, k0)
@assert D ≈ transpose(D) atol=1e-12
println("reciprocity check (D = Dᵀ): OK")

# Self-cell integral has nonzero real and imaginary parts
a_eq   = equivalent_radius(mesh)
D_self = self_cell_integral_2d(k0, a_eq)
@assert abs(real(D_self)) > 0 && abs(imag(D_self)) > 0
println("self-cell integral D_self = ", round(D_self, sigdigits=4),
        "  (a_eq = ", round(a_eq, sigdigits=4), ")")

# Jacobian shape + a single finite-difference spot check
J, _ = jacobian_scattered_field_2d(vr, r_obs)
@assert size(J) == (length(r_obs), mesh.ncells)
p = findfirst(x -> x > 0, vr.chi)          # an interior (dielectric) cell
delta = 1e-7
chi_pert = copy(vr.chi); chi_pert[p] += delta
vr_pert  = solve_vie_2d(mesh, k0, chi_pert, E_inc_fs)
J_fd     = (scattered_field_2d(vr_pert, r_obs) - E_scat_mom) / delta
jac_err  = norm(J[:, p] - J_fd) / norm(J[:, p])
println("Jacobian FD spot-check (cell $p) rel-err = ", round(jac_err, sigdigits=4))

println("DONE")
```

Running this (`julia --project=. /path/to/script.jl`) prints, for example:

```
== 2D VIE (TM) vs analytical Mie series ==
k0 = 20.944 rad/m,  a = 0.03 m,  k0*a = 0.6283,  eps_r = 4.0
grid 8x8  ncells=64  rel-err vs Mie = 0.02177
grid 16x16  ncells=256  rel-err vs Mie = 0.02153
free-space check (χ=0 ⟹ E_total=E_inc): OK
reciprocity check (D = Dᵀ): OK
self-cell integral D_self = 2.718e-5 - 1.405e-5im  (a_eq = 0.004231)
Jacobian FD spot-check (cell 3) rel-err = 5.826e-7
DONE
```

The roughly 2% scattered-field error at these coarse grids reflects the staircase approximation of the curved boundary (Section 9), not a coding error -- the test suite drives the error below 1% on a $40\times40$ grid.

---

## 9. When to Use / Limitations

**Use this 2D VIE when** you need TM scattering from an *inhomogeneous* dielectric region: graded-index profiles, designed permittivity distributions, or any problem where the unknown naturally lives in the volume rather than on a surface. The second-kind Fredholm structure ($Z = I - \text{compact}$) keeps the system well conditioned (the tests check $\mathrm{cond}(Z) < 10^{10}$ for a $25\times25$ case), and the analytic contrast Jacobian makes it a clean target for gradient-based inverse design.

**Be aware of these limitations**, all grounded in the implementation:

- **Staircasing.** Curved scatterers are approximated on a Cartesian grid, so VIE-vs-Mie convergence is **non-monotonic** with refinement (the test explicitly warns about this). The tests assert overall accuracy bands -- coarse grid $<5\%$, fine grid $<1\%$ -- rather than strict monotonicity.
- **Midpoint quadrature.** Off-diagonal $D_{mn}$ uses a single-point rule, accurate only for small cells relative to $\lambda$. Coarse or electrically large cells degrade accuracy.
- **Real contrast only.** `chi` is `Vector{Float64}`; the VIE path models lossless real-$\varepsilon_r$ dielectrics. No lossy or PEC support in the VIE itself (PEC is only in the Mie oracle).
- **Exterior observation/sources.** Observation points and line sources must lie outside the domain; coincidence with a cell center makes the midpoint Green's evaluation singular.
- **Mie truncation.** The oracle uses finite order $N$; boundary conditions and coefficient symmetries hold only to truncation accuracy (tests use $\sim10^{-6}$ / $10^{-12}$ tolerances).

**Validation lives in** `test/test_mom2d.jl`:

| Test set | What it checks |
|----------|----------------|
| `2D Green's function` | $G = (-i/4)H_0^{(2)}(kR)$, symmetry, zero self-sentinel, decay |
| `Self-cell integral` | finite, nonzero real/imag parts; asserts on $a_\text{eq}\le0$, $k\le0$ |
| `VIE assembly and solve` | $Z$ is $25\times25$, $\mathrm{cond}(Z)<10^{10}$; $\chi=0\Rightarrow Z\approx I$ and $E_\text{total}\approx E_\text{inc}$ |
| `Plane wave` / `Line source` | unit-amplitude phase $e^{-ik_0 x}$; line-source decay |
| `MoM vs Mie convergence` | dielectric cylinder, coarse $<5\%$, fine $<1\%$, finer beats coarser |
| `Mie series - PEC / dielectric` | $c_0 = -J_0/H_0^{(2)}$; PEC surface field $<10^{-6}$; $c_{-n}\approx c_n$ |
| `Jacobian accuracy` | $J$ shape, no NaN/Inf, column rel-err $<10^{-4}$ vs finite differences |
| `Reciprocity check` | $D \approx D^\top$ to `atol=1e-13` |

---

## 10. Code Mapping

| Concept | Exported function / type | Source file |
|---------|--------------------------|-------------|
| Mesh, cell centers, equivalent radius | `Mesh2D`, `equivalent_radius`, `Vec2`, `CVec2` | `src/mom2d/Types2D.jl` |
| Solve-result bundle (with cached LU) | `VIEResult2D` | `src/mom2d/Types2D.jl` |
| 2D Green's function | `greens_2d` | `src/mom2d/Greens2D.jl` |
| Analytical self-cell integral | `self_cell_integral_2d` | `src/mom2d/Greens2D.jl` |
| Green's integral matrix $D$ | `assemble_D_matrix` (internal) | `src/mom2d/Greens2D.jl` |
| System matrix $Z = I - k_0^2\,D\,\mathrm{diag}(\chi)$ | `assemble_vie_2d` | `src/mom2d/Assembly2D.jl` |
| LU solve $Z\,E = E_\text{inc}$ | `solve_vie_2d` | `src/mom2d/Assembly2D.jl` |
| Plane-wave / line-source excitation | `planewave_2d`, `linesource_2d` | `src/mom2d/Excitation2D.jl` |
| Observation Green's matrix | `green_obs_matrix` | `src/mom2d/Scatter2D.jl` |
| Scattered field from contrast sources | `scattered_field_2d` | `src/mom2d/Scatter2D.jl` |
| Contrast Jacobian $\partial E_\text{scat}/\partial\chi$ | `jacobian_scattered_field_2d` | `src/mom2d/Scatter2D.jl` |
| 2D Mie coefficients / fields (oracle) | `mie_coefficients_2d`, `mie_scattered_field_2d`, `mie_total_field_2d` | `src/mom2d/Mie2D.jl` |

Full signatures and per-argument tables are in the [2D VIE API reference](../api/vie-2d.md).

---

## 11. Exercises

### Conceptual

1. **Contrast as a source.** Starting from $(\nabla^2 + k_0^2\varepsilon_r)E_z = 0$, re-derive the contrast-source form $(\nabla^2 + k_0^2)E_z = -k_0^2\chi E_z$ and explain why $\chi = \varepsilon_r - 1$ vanishes outside the scatterer.

2. **Time convention.** Show that under $\exp(+i\omega t)$, $H_0^{(2)}(kr)$ represents an outgoing wave while $H_0^{(1)}(kr)$ represents an incoming one. What would change in `greens_2d`, the self-cell term, and the Mie series if the package switched to $\exp(-i\omega t)$?

3. **Why $I -$ compact.** Explain why the discrete VIE has the form $Z = I - k_0^2\,D\,\mathrm{diag}(\chi)$ (a second-kind equation) rather than just $D\,\mathrm{diag}(\chi)\,\mathbf{E} = \cdots$, and why this matters for conditioning.

4. **Self vs. midpoint.** Why can't the midpoint rule be used for the diagonal of $D$? Why is the area-equivalent disk ($\pi a_\text{eq}^2 = A$) a reasonable replacement for a square self-cell?

### Numerical

5. **Convergence sweep.** Extend the worked example to grids $8, 16, 32, 64$ and plot the Mie relative error. Confirm the convergence is non-monotonic (Section 9) and identify the resolution where the error drops below 1%.

6. **Plane-wave free-space identity.** With `chi = zeros(ncells)`, verify that `solve_vie_2d` returns `E_total ≈ E_inc` for several incidence angles `phi_inc`. Then set one cell's `chi` nonzero and observe how `E_total` departs from `E_inc`.

7. **Line-source illumination.** Replace `planewave_2d` with `linesource_2d` at `Vec2(5a, 0.0)` and compute the scattered field on the observation ring. Confirm the source amplitude decays with distance and the solve still satisfies the free-space identity when `chi = 0`.

8. **Jacobian sweep.** Validate `jacobian_scattered_field_2d` against finite differences for all interior cells (not just one), and time the implicit Jacobian against the brute-force $N$-solve finite-difference Jacobian.

### Advanced

9. **Inhomogeneous profile.** Replace the binary cylinder with a graded radial profile $\varepsilon_r(\rho) = 1 + 3(1 - \rho/a)$ inside $\rho \le a$. There is no closed-form Mie reference, but you can still verify the free-space identity, $D$ symmetry, and Jacobian-vs-FD. Discuss how you would build a trusted reference (e.g. a finer VIE or an external FDTD/FEM solver).

10. **Toward inverse design.** Using the contrast Jacobian, set up a least-squares objective $\|E_\text{scat}(\chi) - E_\text{target}\|^2$ over the observation ring and take one gradient-descent step on $\chi$. Outline how the cached `Z_LU` and the implicit-differentiation structure keep the per-iteration cost low.

---

## 12. Chapter Checklist

Before moving on, make sure you can:

- [ ] Derive the Lippmann-Schwinger VIE and identify $\chi = \varepsilon_r - 1$ as the contrast source.
- [ ] State $G_\text{2D} = (-i/4)H_0^{(2)}(kR)$ and justify the second-kind Hankel function from $\exp(+i\omega t)$.
- [ ] Assemble $Z = I - k_0^2\,D\,\mathrm{diag}(\chi)$ with pulse basis + point matching, and explain why $\chi = 0 \Rightarrow Z = I$.
- [ ] Distinguish the off-diagonal midpoint rule from the analytical self-cell integral and the role of $a_\text{eq} = \sqrt{A/\pi}$.
- [ ] Build plane-wave / line-source excitations and recover the scattered field by radiating contrast sources.
- [ ] Compute the contrast Jacobian via implicit differentiation and verify it against finite differences.
- [ ] Run the Mie-series validation and interpret the staircasing-limited accuracy.

---

## 13. Further Reading

1. **Volume integral equations and the contrast source:**
   - Peterson, A. F., Ray, S. L., & Mittra, R. (1998). *Computational Methods for Electromagnetics*. IEEE Press. Pulse-basis VIE discretization and self-cell treatment.
   - Richmond, J. H. (1965). "Scattering by a dielectric cylinder of arbitrary cross section shape." *IEEE Trans. Antennas Propag.* The classic 2D TM VIE with the area-equivalent self-cell.

2. **2D scattering and the Mie series:**
   - Balanis, C. A. (2012). *Advanced Engineering Electromagnetics*, 2nd ed. Wiley. Cylindrical-wave expansions and circular-cylinder scattering.

3. **Inverse scattering / differentiable design:**
   - Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic Scattering Theory*, 3rd ed. Springer. The Lippmann-Schwinger equation as the forward model for inverse problems.

- **DiffMoM.jl source:** `src/mom2d/` for the full implementation; `test/test_mom2d.jl` for the validation suite; the [2D VIE API reference](../api/vie-2d.md) for signatures.

---

*Related: the [Physical Optics](../methods/03-physical-optics.md) chapter covers a complementary high-frequency surface method, and the [Density Topology Optimization](../methods/08-density-topology-optimization.md) chapter builds on the differentiable-design ideas introduced here.*
