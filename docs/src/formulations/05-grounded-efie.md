# Grounded (Half-Space) EFIE via Image Theory

## Purpose

A reflectarray or reconfigurable metasurface almost never floats in free space: it is a thin patterned conductor suspended a small height $h$ above a solid metal backplane. The backplane is essential to the device physics -- it forces all the incident power back into the upper half-space and turns the structure into a phase-only reflector -- but it is also electrically enormous (effectively infinite) and would be ruinously expensive to mesh directly. Image theory solves this exactly for a planar Perfect Electric Conductor (PEC) ground: the ground is removed and replaced by a single mirror copy of the unknown currents, so the discrete problem stays the size of one unit cell.

This chapter derives the grounded periodic EFIE used by `DiffMoM.jl` for a coplanar metasurface a distance $h$ above an infinite PEC ground plane. It builds directly on the free-standing periodic EFIE and Floquet post-processing (see [Periodic EFIE and Floquet Metrics](../methods/07-periodic-efie-and-floquet-metrics.md)), and adds exactly three ingredients: an image impedance block, an image-aware excitation, and image-aware reflection coefficients. Use it whenever you are designing a reflective (grounded) metasurface and want the specular and Floquet reflection budget without meshing the backplane.

---

## Learning Goals

After this chapter, you should be able to:

1. State the image of a horizontal electric current (and its charge) above a PEC ground and explain why a single $-1$ factor handles both potential kernels.
2. Derive the grounded scalar Green's function $G_{\text{grounded}}(\Delta\rho) = G_{\text{per}}(\Delta\rho, 0) - G_{\text{per}}(\Delta\rho, 2h)$ and the matrix identity $Z_{\text{grounded}} = Z_{\text{direct}} - Z_{\text{image}}$.
3. Explain why the image block needs no singularity extraction (source and image are never coincident).
4. Construct the grounded excitation $(1 - e^{-2i k_{z,\text{inc}} h})\,\mathbf{v}_{\text{inc}}$ from the incident wave plus its bare-ground reflection.
5. Assemble the grounded reflection coefficient, including the image-current interference factor and the $(0,0)$ bare-ground background.
6. Verify the two analytic limits (empty cell $\to$ bare ground; full PEC sheet $\to$ total reflection) and the lossless power budget.
7. Avoid the practical pitfalls (positive height, coplanar Bloch-paired mesh, evanescent-order phase).

---

## 1. Image Theory for Horizontal Currents

### 1.1 The Mirror Current and Mirror Charge

Place an infinite PEC ground plane at $z = 0$ and a coplanar sheet of horizontal (tangential to the ground) electric current $\mathbf{J}$ at height $z = h$. The boundary condition on the PEC -- vanishing tangential electric field on the conductor -- is satisfied exactly by removing the ground and adding a single image current

```math
\mathbf{J}_{\text{img}}(\boldsymbol{\rho}, -h) = -\mathbf{J}(\boldsymbol{\rho}, h),
```

a mirror copy at depth $-h$ with the **horizontal** components flipped in sign. The continuity equation $\nabla\cdot\mathbf{J} = -i\omega\rho$ then forces the associated surface charge to image with the **same** $-1$ factor:

```math
\rho_{\text{img}}(\boldsymbol{\rho}, -h) = -\rho(\boldsymbol{\rho}, h).
```

(For a vertical current the signs would differ between current and charge; here we exploit the fact that the metasurface mesh is coplanar, so all currents are horizontal.)

### 1.2 One Sign for Both Potentials

The EFIE couples currents through two potentials: the **vector** potential, weighted by $\mathbf{f}_m\cdot\mathbf{f}_n$, and the **scalar** potential, weighted by $(\nabla\cdot\mathbf{f}_m)(\nabla\cdot\mathbf{f}_n)$. Because the current and its charge both image with $-1$, both potential kernels acquire the image contribution with a **single** $-1$. This is the key simplification: we never have to track separate image signs for the two terms. The implementation states this explicitly in the file header of `src/assembly/GroundedEFIE.jl`:

> the image of a horizontal current $J$ at height $h$ is $-J$ at depth $-h$, and its associated charge images with the same $-1$ factor. Hence both the vector- and scalar-potential kernels acquire the image with a single $-1$.

---

## 2. The Grounded Green's Function

### 2.1 Core Substitution

The free-standing periodic EFIE evaluates both potential integrals with the 2D-periodic (Ewald) Green's function $G_{\text{per}}(\Delta\rho, \Delta z)$ at zero vertical separation, $G_{\text{per}}(\Delta\rho, 0)$. Image theory replaces this scalar kernel everywhere it appears by the **grounded** Green's function

```math
G_{\text{grounded}}(\Delta\rho) = G_{\text{per}}(\Delta\rho, 0) \;-\; G_{\text{per}}(\Delta\rho, 2h),
```

in **both** the $\mathbf{f}\cdot\mathbf{f}$ and $(\nabla\cdot\mathbf{f})(\nabla\cdot\mathbf{f})$ integrals. The first term is the direct (real-current) interaction; the second is the interaction with the mirror current. The vertical separation between a source on the real sheet ($z = z_0$) and a point on the image sheet ($z = z_0 - 2h$) is exactly $2h$, the source–image distance.

### 2.2 Why No Singularity Extraction

The direct term $G_{\text{per}}(\Delta\rho, 0)$ contains the usual $1/R$ singularity when source and test overlap, and is handled by the existing periodic-EFIE singularity treatment. The **image** term $G_{\text{per}}(\Delta\rho, 2h)$ is evaluated at vertical separation $2h > 0$. Real and image points are therefore **never coincident** ($R \ge 2h$), so the image block uses the full periodic Green's function with no singularity extraction. In code this is `_gper_full`, which simply adds the free-space $G_0$ and the Ewald periodic correction:

```julia
@inline function _gper_full(r, rp, k, lattice)
    R  = norm(r - rp)
    g0 = exp(-im * k * R) / (4π * R)        # exp(+iωt), G0 = exp(-ikR)/(4πR)
    return g0 + greens_periodic_correction(r, rp, k, lattice)
end
```

The image source quadrature points are the real ones shifted straight down by $2h$: `shift = SVector(0, 0, two_h)`.

!!! warning "Do not reuse `_gper_full` for self-interactions"
    `_gper_full` is valid only because the real and image points are separated by $2h > 0$. It performs **no** singularity extraction and must not be used for coincident or self-interaction terms.

---

## 3. The Matrix Identity

### 3.1 Direct minus Image

Discretizing with the same RWG basis as the free-standing problem, the grounded impedance matrix splits cleanly into the existing free-standing matrix and an image block:

```math
\boxed{\,Z_{\text{grounded}} = Z_{\text{direct}} - Z_{\text{image}}\,}
```

- $Z_{\text{direct}}$ is the existing coplanar periodic EFIE, assembled by `assemble_Z_efie_periodic` -- unchanged.
- $Z_{\text{image}}$ is the interaction of each real basis function with the **mirror** basis functions at depth $2h$, assembled by `_assemble_periodic_image_block`.

`assemble_Z_efie_grounded` simply forms both and subtracts:

```julia
function assemble_Z_efie_grounded(mesh, rwg, k, lattice; height, quad_order=3, eta0=376.730313668)
    height > 0 || throw(ArgumentError("ground-plane height must be positive (got $height)"))
    Z_direct = assemble_Z_efie_periodic(mesh, rwg, k, lattice; quad_order, eta0)
    Z_image  = _assemble_periodic_image_block(mesh, rwg, k, lattice, 2 * Float64(height); quad_order, eta0)
    return Z_direct - Z_image
end
```

Note the image block is evaluated at `two_h = 2 * height`, the full source-to-image distance.

### 3.2 The Image-Block Entry Kernel

Each image-block entry is the standard mixed-potential EFIE entry, evaluated with $G = G_{\text{per,full}}$ at $\Delta z = 2h$:

```math
Z_{\text{img}}[m,n] = -i\,\omega\mu_0 \iint \!\left[\, \mathbf{f}_m\cdot\mathbf{f}_n\, G \;-\; \frac{1}{k^2}(\nabla\cdot\mathbf{f}_m)(\nabla\cdot\mathbf{f}_n)\, G \,\right] dS\, dS'.
```

In the implementation $\omega\mu_0 = k\,\eta_0$ (with default $\eta_0 = 376.730313668\,\Omega$), $1/k^2$ is the inverse-square factor on the charge term, and the divergence term uses $\overline{\nabla\cdot\mathbf{f}_m}\,(\nabla\cdot\mathbf{f}_n)$. The same $-i\omega\mu_0\big[\iint\mathbf{f}\cdot\mathbf{f}\,G - \tfrac{1}{k^2}\iint(\nabla\cdot\mathbf{f})(\nabla\cdot\mathbf{f})\,G\big]$ form is the documented entry kernel of the free-standing periodic EFIE, so $Z_{\text{direct}}$ and $Z_{\text{image}}$ are dimensionally and structurally identical -- only the Green's-function argument ($\Delta z = 0$ vs. $\Delta z = 2h$) differs.

!!! note "Fast symmetric path"
    At normal incidence ($k_{x,\text{bloch}} = k_{y,\text{bloch}} = 0$) the kernel cache is reciprocal, so only the upper triangle of the $G$-cache is computed and mirrored. Oblique incidence (nonzero Bloch phase) uses the full sweep.

---

## 4. Convention and the Image Phase

`DiffMoM.jl` uses the $e^{+i\omega t}$ time convention with $G_0 = e^{-ikR}/(4\pi R)$. This is stated explicitly in `src/basis/PeriodicGreens.jl` ("`Convention: exp(+iωt), G_0 = exp(-ikR)/(4πR)`") and used in `_gper_full` (`g0 = exp(-im*k*R)/(4π*R)`).

A consequence threads through everything below: a wave that travels the extra round-trip distance $2h$ down to the ground and back picks up a phase delay $e^{-2i k_z h}$ with a **minus** sign. This same factor appears in three places, always with the minus sign:

- the grounded **excitation**: $\bigl(1 - e^{-2i k_{z,\text{inc}} h}\bigr)$,
- the per-mode **image-current factor**: $\bigl(1 - e^{-2i\,\mathrm{Re}(k_{z,mn})\, h}\bigr)$,
- the **bare-ground background**: $-e^{-2i k_{z,\text{inc}} h}$ (applied only to the $(0,0)$ order).

The Bloch/Floquet context is the periodic one: $k_{x,\text{bloch}} = k\sin\theta\cos\phi$, $k_{y,\text{bloch}} = k\sin\theta\sin\phi$, the grating wavenumbers are $k_{x,mn} = k_{x,\text{bloch}} + 2\pi m/d_x$ and $k_{y,mn} = k_{y,\text{bloch}} + 2\pi n/d_y$, and the incident vertical wavenumber of the specular order is

```math
k_{z,\text{inc}} = \sqrt{\max\!\bigl(k^2 - k_{x,\text{bloch}}^2 - k_{y,\text{bloch}}^2,\, 0\bigr)} = k\cos\theta_{\text{inc}},
```

computed by `_kz_inc(k, lattice)`.

---

## 5. The Grounded Excitation

### 5.1 Incident Wave Plus Its Bare-Ground Reflection

In a half-space the metasurface is not driven by the incident wave alone: it also sees the wave reflected by the bare ground. Superposing the down-going incident wave and its up-going ground reflection at the metasurface plane $z = 0$ produces a standing-wave amplitude that scales the free-standing excitation by $\bigl(1 - e^{-2i k_{z,\text{inc}} h}\bigr)$:

```math
\mathbf{v}_{\text{grounded}} = \bigl(1 - e^{-2i k_{z,\text{inc}} h}\bigr)\,\mathbf{v}_{\text{inc}},
```

where $\mathbf{v}_{\text{inc}}$ is the ordinary free-standing excitation from `assemble_excitation`. This is exactly `assemble_excitation_grounded`:

```julia
function assemble_excitation_grounded(mesh, rwg, pw, k, lattice; height, quad_order=3)
    v_inc  = assemble_excitation(mesh, rwg, pw; quad_order)
    factor = 1 - exp(-2im * _kz_inc(k, lattice) * height)
    return factor .* v_inc
end
```

### 5.2 The Vanishing-Drive Heights

The factor $\bigl(1 - e^{-2i k_{z,\text{inc}} h}\bigr)$ vanishes when $2 k_{z,\text{inc}} h$ is a multiple of $2\pi$ -- for example $h = \lambda/2$ at normal incidence. Physically the incident and ground-reflected fields cancel at the sheet (a field node), so the sheet is undriven. The strongest drive occurs at the quarter-wave height $h = \lambda/4$, where $2 k_{z,\text{inc}} h = \pi$ and the factor is $1 - e^{-i\pi} = 2$.

---

## 6. Grounded Reflection Coefficients

### 6.1 The Free-Standing Per-Mode Coefficient

The post-processing starts from the free-standing per-mode reflection coefficient (see [Periodic EFIE and Floquet Metrics](../methods/07-periodic-efie-and-floquet-metrics.md)):

```math
R_{mn} = -\frac{\eta_0 k}{2\,k_{z,mn}\,E_0}\,\bigl(\hat{\mathbf{e}}_{\text{pol}}\cdot \tilde{\mathbf{J}}_{mn}\bigr),
\qquad
\tilde{\mathbf{J}}_{mn} = \frac{1}{A}\int_{\text{cell}} \mathbf{J}(\mathbf{r}')\, e^{i\,\boldsymbol{\kappa}_t\cdot\mathbf{r}'}\, dS',
```

the mode-transverse co-polar projection of the current Fourier coefficient. (Sanity: a PEC plate at normal incidence gives $R_{00} = -1$.) We write $R_{mn}^{\text{cur}}$ for this current contribution.

### 6.2 The Grounded Coefficient

The grounded reflection coefficient adds the mirror current and, for the specular order only, the bare-ground specular background:

```math
\boxed{\,R_{mn}^{\text{grounded}} = R_{mn}^{\text{cur}}\,\bigl(1 - e^{-2i k_{z,mn} h}\bigr) \;-\; \delta_{mn,(0,0)}\, e^{-2i k_{z,\text{inc}} h}\,}
```

The first factor is the image-current **interference**: the field of the real current plus the phase-delayed field of its mirror. The second term is the field that the **bare** ground would reflect specularly, which exists independently of the metasurface and is therefore added only to the $(0,0)$ order:

```julia
R_g[i] = R_cur[i] * (1 - exp(-2im * real(m.kz) * h))
if m.m == 0 && m.n == 0
    R_g[i] -= exp(-2im * kzi * h)
end
```

### 6.3 Evanescent Orders Use `real(kz)`

Evanescent Floquet orders store $k_z = i\beta$ with positive imaginary part. Naively forming $e^{-2i k_z h} = e^{2\beta h}$ would **overflow**, and since the current contribution $R_{mn}^{\text{cur}}$ is exactly $0$ for those orders, the product would be $0\cdot\infty = \mathrm{NaN}$. The code deliberately uses $\mathrm{Re}(k_z)$ in the image phase so the phase delay is governed by the real vertical wavenumber, consistent between the scalar and vector grounded routines.

!!! warning "Mirror `real(kz)` in any hand-rolled post-processing"
    If you recompute the grounded per-mode map yourself, use `real(m.kz)` in $e^{-2i k_z h}$, never the complex `m.kz`. Using the complex value overflows on evanescent orders and produces `NaN`.

---

## 7. Limiting Behaviors

Two limits make the formula auditable, and both are exercised numerically in the worked example below.

**(a) Empty cell (bare ground).** With zero current, $R_{mn}^{\text{cur}} = 0$, so only the background survives:

```math
R_{00} = -e^{-2i k_{z,\text{inc}} h}, \qquad |R_{00}| = 1.
```

At normal incidence with $h = \lambda/4$ this is $-e^{-2i k \lambda/4} = -e^{-i\pi} = +1$. A bare ground reflects all power; the sign/phase tracks the round-trip delay.

**(b) Full PEC sheet at $z = 0$.** A solid conductor on the ground reflects everything regardless of height:

```math
R_{00} = -1 \quad \text{for any } h.
```

This is the grounded counterpart of the free-standing PEC-plate identity $R_{00} = -1$ at normal incidence.

---

## 8. The Vector Form and Energy Budget

For a power budget you must keep **both** transverse polarizations of every propagating order, not just the co-polar projection. `reflection_coefficient_vectors_grounded` does this: it calls the free-standing vector routine, applies the **same** image factor $\bigl(1 - e^{-2i\,\mathrm{Re}(k_z)\,h}\bigr)$, and subtracts the $(0,0)$ bare-ground background **projected onto the mode-transverse plane of `pol`** (skipped when that projection is undefined). Passing the result to `reflected_power_fractions` gives the per-order reflected-power fraction

```math
p_i = |\mathbf{R}_i^{\text{vec}}|^2\,\frac{\mathrm{Re}(k_{z,i})}{k},
```

whose total should be $\approx 1$ for a lossless (purely reactive) loaded surface -- no power is transmitted past the ground and none is absorbed.

---

## Worked Example

The following script is self-contained and runs in seconds with
`julia --project=/Users/jake/DiffMoM.jl <file>.jl`. It assembles the grounded
EFIE on a tiny $0.5\lambda$ unit cell at 10 GHz, performs a single forward solve,
and checks all three observables against analytic values: the full-PEC-sheet
limit $|R_{00}| = 1$, the empty-cell bare-ground background $R_{00} = -e^{-2i k_{z,\text{inc}} h}$,
the lossless power budget $\approx 1$, and the positive-height guard.

```julia
# Grounded (Half-Space) EFIE via Image Theory — minimal forward-solve demo.
#
# A coplanar periodic metasurface sits a height h above an infinite PEC ground
# plane. Image theory replaces the ground by the mirror currents (-J at depth -h):
#
#     Z_grounded = Z_direct - Z_image
#
# The structure is driven by the incident plane wave PLUS its bare-ground
# reflection (factor 1 - exp(-2i*kz_inc*h)). We solve once and read off the
# specular Floquet reflection coefficient R00.
#
# Convention check (PeriodicGreens.jl line 23): exp(+iωt), G0 = exp(-ikR)/(4πR).

using DiffMoM, LinearAlgebra, StaticArrays, Printf

const C0 = 2.99792458e8
freq = 10e9
lam  = C0 / freq
k    = 2π / lam
eta0 = 376.730313668

dxc = 0.5 * lam        # sub-wavelength unit cell (only the (0,0) order propagates)
Nx  = 6                # tiny mesh -> runs in seconds
h   = lam / 4          # ground-plane height

# --- Geometry: coplanar unit-cell plate + periodic lattice + Bloch-paired RWG ---
mesh = make_rect_plate(dxc, dxc, Nx, Nx)
lat  = PeriodicLattice(dxc, dxc, 0.0, 0.0, k)   # normal incidence: kx=ky=0
rwg  = build_rwg_periodic(mesh, lat; precheck=true, allow_boundary=true, require_closed=false)
N    = rwg.nedges
@printf("mesh: %d triangles, %d RWG edges; cell = %.3fλ, h = λ/4\n",
        ntriangles(mesh), N, dxc / lam)

# --- (1) Bare-ground sanity check: a FULL PEC sheet must give |R00| = 1 ---
# Driving the grounded EFIE with the bare-ground excitation and solving yields
# the PEC-sheet-on-ground response; |R00| should be 1 (lossless full reflector).
Zg = assemble_Z_efie_grounded(mesh, rwg, k, lat; height=h)
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
v  = Vector{ComplexF64}(assemble_excitation_grounded(mesh, rwg, pw, k, lat; height=h))

I  = Zg \ v          # single forward solve

modes, R_g = reflection_coefficients_grounded(mesh, rwg, I, k, lat;
                 height=h, N_orders=1, E0=1.0, pol=SVector(1.0, 0.0, 0.0))
i00 = findfirst(m -> m.m == 0 && m.n == 0, modes)
@printf("full PEC sheet on ground:  R00 = %+.4f %+.4fi  |R00| = %.4f\n",
        real(R_g[i00]), imag(R_g[i00]), abs(R_g[i00]))

# --- (2) Bare-ground limit from an EMPTY cell (zero current) ---
# reflection_coefficients_grounded with I = 0 reproduces the analytic bare-ground
# specular background  R00 = -exp(-2i*kz_inc*h),  with |R00| = 1.
modes0, R0 = reflection_coefficients_grounded(mesh, rwg, zeros(ComplexF64, N), k, lat;
                 height=h, N_orders=1, E0=1.0, pol=SVector(1.0, 0.0, 0.0))
j00 = findfirst(m -> m.m == 0 && m.n == 0, modes0)
kz_inc = k                              # normal incidence: kz_inc = k cosθ = k
R00_analytic = -exp(-2im * kz_inc * h)  # -exp(-i*k*λ/2) = -exp(-iπ) = +1
@printf("empty cell (I=0):          R00 = %+.4f %+.4fi  (analytic %+.4f %+.4fi)\n",
        real(R0[j00]), imag(R0[j00]), real(R00_analytic), imag(R00_analytic))

# --- (3) Vector form + reflected-power budget (lossless check) ---
modesv, Rv = reflection_coefficient_vectors_grounded(mesh, rwg, I, k, lat;
                 height=h, N_orders=1, E0=1.0, pol=SVector(1.0, 0.0, 0.0))
budget = sum(reflected_power_fractions(modesv, Rv, k))
@printf("total reflected power budget (should be ~1) = %.4f\n", budget)

# --- (4) Positive height is required ---
ok = false
try
    assemble_Z_efie_grounded(mesh, rwg, k, lat; height=-1.0)
catch e
    global ok = e isa ArgumentError
end
@printf("negative height rejected with ArgumentError: %s\n", ok)

println("done.")
```

Running this prints (numbers reproduced from an actual run):

```text
mesh: 72 triangles, 108 RWG edges; cell = 0.500λ, h = λ/4
full PEC sheet on ground:  R00 = -1.0000 +0.0040i  |R00| = 1.0000
empty cell (I=0):          R00 = +1.0000 +0.0000i  (analytic +1.0000 +0.0000i)
total reflected power budget (should be ~1) = 1.0000
negative height rejected with ArgumentError: true
done.
```

The full PEC sheet on ground gives $|R_{00}| = 1.0000$; the empty cell reproduces the analytic bare-ground $R_{00} = -e^{-2i k\lambda/4} = +1.0000$; the total reflected-power budget is $1.0000$; and a negative height is rejected with `ArgumentError`.

---

## Validation

The grounded EFIE has a dedicated validation testset, `"B: Grounded metasurface (image theory)"` in `test/test_periodic_topology.jl` (a $0.5\lambda$ cell, $N_x = 6$, at 10 GHz):

1. **Full reflection.** A full PEC sheet on the ground reflects fully: $|R_{00}| \approx 1.0$ (`atol=2e-3`) for both $h = \lambda/8$ and $h = \lambda/4$.
2. **Transmission-line oracle.** A uniform reactive sheet matches the **exact** transmission-line solution $R_{00} = (Z_{\text{in}} - \eta_0)/(Z_{\text{in}} + \eta_0)$ with $Z_{\text{in}} = Z_s \,\|\, (j\eta_0\tan kh)$, tested at $h = \lambda/8$, $\lambda/4$, $3\lambda/8$ (`atol=2e-3`). This is an independent closed-form oracle.
3. **Lossless power conservation.** The scalar Floquet sum $\approx 1.0$ (`atol=3e-3`) and the full vector budget `sum(reflected_power_fractions) ≈ 1.0` (`atol=3e-3`); the scalar and vector routines return identical mode lists.
4. **Guard.** A negative height raises `ArgumentError`.

End-to-end usage lives in `examples/21_grounded_rcs_demo.jl` (a grounded RCS topology-optimization demo on a $1.2\lambda$ cell at $h = \lambda/4$, $N_x = 14$). The scripts in `examples/grounded_rcs/` (`framework_energy_honest.jl`, `framework_pixel_design.jl`) recompute the grounded per-mode maps independently and cross-check them against `reflection_coefficient_vectors_grounded` / `reflected_power_fractions`, serving as an independent oracle for the reflection formulas.

---

## When to Use / Limitations

Use the grounded EFIE when modeling a coplanar periodic conductor a known height above an **infinite planar PEC** ground (reflectarrays, reconfigurable intelligent surfaces, frequency-selective reflectors). It is exact for that geometry and costs no more unknowns than the free-standing periodic problem -- the ground never enters the mesh.

It is **not** applicable to: non-planar or finite-extent grounds; dielectric or lossy (non-PEC) backplanes; vertical currents (the single-$-1$ image identity assumes horizontal currents); or transmissive (no-ground) structures, for which the free-standing periodic EFIE applies directly.

**Pitfalls to avoid:**

- **Positive height required.** `assemble_Z_efie_grounded` throws `ArgumentError` for `height <= 0`.
- **Coplanar mesh.** The mesh must lie at $z = \text{const}$ (the internal `assemble_Z_efie_periodic` and the reflection routines assert max z-spread $\le 10^{-12}$). The mesh sits at $z = 0$; the ground/image height is supplied via the `height` keyword, **not** by moving the mesh.
- **Bloch-paired RWG for boundary-touching cells.** If conductor edges lie on the unit-cell boundary, the RWG must be built with `build_rwg_periodic` (the demo uses `allow_boundary=true, require_closed=false`).
- **No self-interaction via `_gper_full`.** The image block's full Green's function is valid only because real and image points are separated by $2h > 0$.
- **Evanescent phase uses `real(kz)`.** Mirror this in any hand-rolled post-processing to avoid `NaN`.
- **Bare-ground background is $(0,0)$-only.** Higher Floquet orders get only the image-current factor.

---

## Code Mapping

| Concept | Exported function / type | Source file |
|---------|--------------------------|-------------|
| Grounded impedance matrix $Z_{\text{direct}} - Z_{\text{image}}$ | `assemble_Z_efie_grounded` | `src/assembly/GroundedEFIE.jl` |
| Grounded excitation $(1 - e^{-2i k_{z,\text{inc}} h})\,\mathbf{v}_{\text{inc}}$ | `assemble_excitation_grounded` | `src/assembly/GroundedEFIE.jl` |
| Scalar grounded reflection coefficient | `reflection_coefficients_grounded` | `src/assembly/GroundedEFIE.jl` |
| Vector grounded reflection (energy budget) | `reflection_coefficient_vectors_grounded` | `src/assembly/GroundedEFIE.jl` |
| Image block (internal) | `_assemble_periodic_image_block`, `_gper_full`, `_kz_inc` | `src/assembly/GroundedEFIE.jl` |
| Free-standing periodic EFIE ($Z_{\text{direct}}$) | `assemble_Z_efie_periodic` | `src/assembly/PeriodicEFIE.jl` |
| Periodic Green's function / lattice | `greens_periodic_correction`, `PeriodicLattice` | `src/basis/PeriodicGreens.jl` |
| Free-standing Floquet reflection | `reflection_coefficients`, `reflection_coefficient_vectors` | `src/postprocessing/PeriodicMetrics.jl` |
| Reflected-power budget, Floquet modes | `reflected_power_fractions`, `floquet_modes`, `FloquetMode` | `src/postprocessing/PeriodicMetrics.jl` |
| Bloch-paired RWG basis | `build_rwg_periodic` | `src/basis/` |
| Geometry / plane wave | `make_rect_plate`, `make_plane_wave`, `Vec3` | `src/geometry/Mesh.jl`, `src/assembly/Excitation.jl`, `src/Types.jl` |

See the [Grounded EFIE API page](../api/grounded-efie.md) for full signatures, parameter tables, and per-function examples.

---

## Exercises

### Conceptual

1. **Image signs.** Explain why a *vertical* electric current does **not** image with a single $-1$ for both potential kernels, and why the coplanar-mesh restriction lets the grounded EFIE use one sign. (Hint: consider the direction of the mirror current and the sign of the imaged charge.)

2. **Vanishing drive.** From the excitation factor $\bigl(1 - e^{-2i k_{z,\text{inc}} h}\bigr)$, find all heights $h$ at normal incidence for which the metasurface is undriven. Which height maximizes the drive, and what is the maximum factor?

3. **Background placement.** Why is the bare-ground specular background $-e^{-2i k_{z,\text{inc}} h}$ added only to the $(0,0)$ order and not to higher Floquet orders?

### Numerical

4. **Height sweep.** Modify the worked example to sweep $h \in \{\lambda/8, \lambda/4, 3\lambda/8, \lambda/2\}$ for a *loaded* (non-PEC) sheet and plot $|R_{00}|$ and $\arg R_{00}$ versus $h$. Confirm the drive vanishes near $h = \lambda/2$.

5. **Transmission-line cross-check.** Reproduce validation test (2): extract the sheet impedance $Z_s$ from a free-standing solve, then compare `reflection_coefficients_grounded` against $R_{00} = (Z_{\text{in}} - \eta_0)/(Z_{\text{in}} + \eta_0)$ with $Z_{\text{in}} = Z_s \,\|\, (j\eta_0\tan kh)$ at several heights.

6. **Power budget under refinement.** For a reactive loaded surface, verify that `sum(reflected_power_fractions(...))` approaches 1 as you refine $N_x$, and report the residual at $N_x = 6, 10, 14$.

### Advanced

7. **Oblique incidence.** Build a `PeriodicLattice` with a nonzero incident angle and confirm the code takes the full (non-symmetric) $G$-cache path. Check that the empty-cell limit becomes $R_{00} = -e^{-2i k_{z,\text{inc}} h}$ with $k_{z,\text{inc}} = k\cos\theta_{\text{inc}} < k$.

8. **Independent oracle.** Following `examples/grounded_rcs/framework_energy_honest.jl`, recompute the grounded vector reflection map by hand (image factor $1 - e^{-2i\,\mathrm{Re}(k_z) h}$ and $(0,0)$ background) and verify it matches `reflection_coefficient_vectors_grounded` to machine precision -- including the correct handling of evanescent orders via `real(kz)`.

---

*Related: [Periodic EFIE and Floquet Metrics](../methods/07-periodic-efie-and-floquet-metrics.md) covers the free-standing periodic assembly and Floquet post-processing this chapter builds on. [Density-Based Topology Optimization](../methods/08-density-topology-optimization.md) shows how to optimize the loaded sheet that sits above the ground.*
