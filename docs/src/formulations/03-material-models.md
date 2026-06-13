# Material Models: Dispersive and Anisotropic Media

## Purpose

The 3D volume solver (discrete-dipole / volume MoM) in `DiffMoM.jl` scatters waves off *material* bodies rather than perfect conductors. To do that it needs a constitutive description of each voxel: how the relative permittivity $\varepsilon_r$ and relative permeability $\mu_r$ respond to the field. Real materials are rarely a single lossless scalar. They can be lossy, anisotropic (direction-dependent), dispersive (frequency-dependent), magnetodielectric, or even bianisotropic (electric and magnetic responses coupled). This chapter develops the small algebra of constitutive models that captures all of these cases.

The material models are deliberately *pure*: each one is a tiny immutable struct plus an evaluator that returns a number, a 3-vector, a $3\times3$ tensor, or a $6\times6$ matrix at a requested frequency. They carry no geometry and no solver state. You evaluate a model at your operating frequency, then hand the resulting array to `solve_dda_3d`. This separation keeps the physics of *what the material is* cleanly apart from the numerics of *how the field is solved*, and it lets a dispersive metal, an anisotropic crystal, and a coupled metamaterial all flow through exactly the same solver path. This chapter derives the constitutive forms, pins down the sign and unit conventions (which are easy to get wrong), and shows the end-to-end coupling into the volume solver.

---

## Learning Goals

After this chapter, you should be able to:

1. State the package's $e^{+i\omega t}$ time convention and the passivity sign condition $\operatorname{Im}\varepsilon_r \le 0$, $\operatorname{Im}\mu_r \le 0$ that follows from it.
2. Distinguish isotropic, diagonal-anisotropic, and full-tensor constitutive models, and state the tensor passivity criterion in terms of the anti-Hermitian loss matrix.
3. Derive and evaluate the Drude, Lorentz, and Debye dispersion laws, getting the angular-frequency factors of $2\pi$ and the loss-term sign right.
4. Combine permittivity and permeability into a magnetodielectric medium, and a $6\times6$ constitutive matrix into a bianisotropic medium.
5. Evaluate a model with `material_epsr_3d` / `material_mur_3d` and feed the result into `solve_dda_3d` through the Clausius-Mossotti polarizability.
6. Recognize when a model construction will be *rejected* by the default passivity checks, and when to deliberately pass `passive=false` (gain media, asymmetric tensors).

---

## 1. Time Convention and Passivity

### 1.1 The $e^{+i\omega t}$ Convention

Every formula in this subsystem assumes the harmonic time dependence

```math
\text{physical field} = \operatorname{Re}\!\left[\,\tilde{\mathbf{E}}(\mathbf{r})\, e^{+i\omega t}\,\right],
```

which is stated in the source header of `MaterialModels3D.jl`. This is a *choice*: many physics texts use $e^{-i\omega t}$ instead, and the two conventions are complex conjugates of each other. Mixing them flips the sign of every imaginary part, so the convention must be nailed down before any loss term is written.

The companion radiated-field convention in the volume solver is $e^{-ikR}$: the electric dipole dyadic carries $e^{-ikR}/(4\pi)$ and the incident plane wave carries $e^{-i\mathbf{k}\cdot\mathbf{r}}$ (`DDA3D.jl`). An outgoing wave $\propto e^{i(\omega t - kR)}$ is exactly the pairing $e^{+i\omega t}\,e^{-ikR}$, so the two conventions are mutually consistent.

### 1.2 The Passivity Sign Condition

A passive medium absorbs energy; it never supplies it. Under $e^{+i\omega t}$, the time-averaged power dissipated per unit volume in a dielectric is proportional to $-\omega\,\varepsilon_0\operatorname{Im}\varepsilon_r\,|\mathbf{E}|^2$. For this to be non-negative (dissipation, not gain) we need

```math
\operatorname{Im}\varepsilon_r \le 0, \qquad \operatorname{Im}\mu_r \le 0.
```

This is the opposite sign to what you would write under $e^{-i\omega t}$, where lossy media have $\operatorname{Im}\varepsilon_r \ge 0$. The code enforces the $e^{+i\omega t}$ version directly. The scalar check is

```julia
imag(z) <= _PASSIVITY_TOL_3D    # _validate_passive_scalar_3d
```

with the small positive tolerance

```math
\texttt{\_PASSIVITY\_TOL\_3D} = 100\,\varepsilon_{\text{mach}}, \qquad \varepsilon_{\text{mach}} = \texttt{eps(Float64)} \approx 2.2\times10^{-16}.
```

Because the comparison is $\le$ a tiny positive number, an exactly lossless medium ($\operatorname{Im}\varepsilon_r = 0$) passes, while any physically meaningful gain ($\operatorname{Im}\varepsilon_r > 0$) is rejected. Passivity is checked **by default** in every constructor and every dispersive evaluator; you opt out with `passive=false`.

---

## 2. Scalar, Diagonal, and Tensor Constitutive Models

### 2.1 Isotropic Media

The simplest constitutive relation is a single complex scalar applied to all three field components:

```math
\mathbf{D} = \varepsilon_0\,\varepsilon_r\,\mathbf{E}, \qquad \varepsilon_r \in \mathbb{C}.
```

`IsotropicMaterial3D(eps_r)` stores this scalar, and the analogous `IsotropicPermeability3D(mu_r)` stores a relative permeability. Both validate that the value is finite and, by default, passive.

### 2.2 Diagonal Anisotropy

A uniaxial or biaxial crystal aligned to the coordinate axes has a diagonal permittivity tensor:

```math
\boldsymbol{\varepsilon}_r = \begin{pmatrix} \varepsilon_x & 0 & 0 \\ 0 & \varepsilon_y & 0 \\ 0 & 0 & \varepsilon_z \end{pmatrix}, \qquad D_a = \varepsilon_0\,\varepsilon_a\,E_a.
```

`DiagonalAnisotropicMaterial3D((eps_x, eps_y, eps_z))` stores the three principal-axis values as an `SVector{3}`. Passivity is checked componentwise: $\operatorname{Im}\varepsilon_a \le 0$ for each axis $a$.

### 2.3 Full Tensors and the Anti-Hermitian Loss Matrix

A general anisotropic medium with off-axis coupling needs the full $3\times3$ tensor

```math
\mathbf{D} = \varepsilon_0\,\boldsymbol{\varepsilon}_r\,\mathbf{E}, \qquad \boldsymbol{\varepsilon}_r \in \mathbb{C}^{3\times3}.
```

Passivity is now more subtle than "every imaginary part $\le 0$." Decompose the tensor into Hermitian and anti-Hermitian parts. The anti-Hermitian part governs power exchange. Define the **loss matrix**

```math
\mathbf{L} = \frac{\boldsymbol{\varepsilon}_r - \boldsymbol{\varepsilon}_r^{\dagger}}{2i},
```

which is Hermitian by construction. The medium is passive if and only if $\mathbf{L}$ is negative semidefinite:

```math
\max\operatorname{eig}\!\big(\mathbf{L}\big) \le \texttt{\_PASSIVITY\_TOL\_3D}.
```

`TensorAnisotropicMaterial3D(eps_r)` (and `TensorPermeability3D`) coerces the input to an `SMatrix{3,3}` and runs this eigenvalue test in `_validate_passive_tensor_3d`.

This criterion has a consequence worth highlighting: a tensor with **real but asymmetric** off-diagonals can fail passivity even with zero imaginary parts. For example,

```julia
TensorAnisotropicMaterial3D(ComplexF64[2.5 0.12 0; 0.04 1.8 0; 0 0 1.0])
```

throws, because $\boldsymbol{\varepsilon}_r \ne \boldsymbol{\varepsilon}_r^\dagger$ makes $\mathbf{L}$ nonzero and not negative semidefinite. Such a tensor is mathematically legitimate as a solver input; you build it with `passive=false`.

---

## 3. Dispersive Permittivity Models

Static models return the same value regardless of the frequency argument (it is only validated as finite and nonnegative). Dispersive models interpret that argument as a frequency in Hz and return a frequency-dependent $\varepsilon_r(\omega)$. All three classic dispersion laws are built on the angular frequency

```math
\omega = 2\pi f, \qquad f \text{ in Hz}.
```

Every *rate* parameter the user supplies is in Hz and is internally multiplied by $2\pi$ to become angular: the plasma frequency, resonance frequency, and damping rate. The Debye relaxation time $\tau$ is the one exception -- it is already in seconds and enters as $\omega\tau$ with no extra $2\pi$. Mixing Hz and rad/s shifts a resonance by a factor of $2\pi$.

### 3.1 Drude (Free-Electron / Plasma) Response

A gas of free electrons with number density driving plasma frequency $\omega_p$ and collision rate $\gamma$ gives the Drude permittivity. Under $e^{+i\omega t}$ the code uses

```math
\varepsilon_r(\omega) = \varepsilon_\infty - \frac{\omega_p^2}{\omega^2 - i\gamma\omega},
\qquad \omega = 2\pi f,\ \ \omega_p = 2\pi f_p,\ \ \gamma = 2\pi\gamma_{\text{Hz}}.
```

The sign of the $-i\gamma\omega$ term in the denominator is chosen so that the resulting $\operatorname{Im}\varepsilon_r \le 0$ -- passive loss. Below the plasma frequency the real part goes negative, reproducing the metal-like screening response. As a concrete check, `drude_epsr_3d(2.0e14; eps_inf=1.0, plasma_freq_hz=1.0e15, gamma_hz=1.0e13)` returns $\varepsilon_r \approx -23.94 - 1.25i$ (negative real part, negative imaginary part).

`DrudePermittivity3D(eps_inf, plasma_freq_hz, gamma_hz)` stores the parameters; the model evaluator delegates to the standalone `drude_epsr_3d`. Because $\omega$ appears in the denominator, Drude evaluation requires $f > 0$ strictly (the DC limit $\omega\to0$ would make the denominator vanish and is rejected rather than returning `Inf`).

### 3.2 Lorentz (Bound-Charge Resonance) Response

A bound charge behaves as a damped harmonic oscillator with resonance $\omega_0$. A single Lorentz oscillator of strength $s$ contributes

```math
\varepsilon_r(\omega) = \varepsilon_\infty + \frac{s\,\omega_0^2}{\omega_0^2 - \omega^2 + i\gamma\omega},
\qquad \omega_0 = 2\pi f_0.
```

Note the $+i\gamma\omega$ here (versus $-i\gamma\omega$ in Drude): the different placement of $\omega^2$ in the denominator is what keeps the loss passive under the same convention. A passive oscillator additionally requires a non-negative real strength, $\operatorname{Re} s \ge 0$, checked at construction time. The real part rises and then dips through the resonance (anomalous dispersion), while the imaginary part shows the characteristic absorption peak near $\omega = \omega_0$. `LorentzPermittivity3D(eps_inf, strength, resonance_freq_hz, gamma_hz)` accepts $f \ge 0$ (the DC value is well defined) but requires $f_0 > 0$.

### 3.3 Debye (Orientational Relaxation) Response

Polar molecules (water, for instance) relax orientationally with a single time constant $\tau$, giving the Debye law

```math
\varepsilon_r(\omega) = \varepsilon_\infty + \frac{\varepsilon_s - \varepsilon_\infty}{1 + i\omega\tau},
\qquad \omega = 2\pi f,\ \ \tau \text{ in seconds}.
```

Here $\varepsilon_s$ is the static (zero-frequency) permittivity and $\varepsilon_\infty$ the high-frequency limit. The $+i\omega\tau$ denominator again yields $\operatorname{Im}\varepsilon_r \le 0$. A passive relaxation requires $\operatorname{Re}(\varepsilon_s - \varepsilon_\infty) \ge 0$ (the permittivity must drop, not rise, with frequency), checked at construction. `DebyePermittivity3D(eps_static, eps_inf, tau_s)` accepts $f \ge 0$ and requires $\tau > 0$.

---

## 4. Magnetodielectric and Bianisotropic Media

### 4.1 Pairing Permittivity with Permeability

A magnetodielectric medium has both a nontrivial $\varepsilon_r$ and a nontrivial $\mu_r$. `MagneticMaterial3D(eps_model, mu_model)` simply pairs any permittivity model with any permeability model:

```math
\mathbf{D} = \varepsilon_0\,\varepsilon_r(\omega)\,\mathbf{E}, \qquad \mathbf{B} = \mu_0\,\mu_r(\omega)\,\mathbf{H}.
```

The evaluators delegate: `material_epsr_3d` on a `MagneticMaterial3D` calls into its stored `eps_model`, and `material_mur_3d` calls into its `mu_model`. Either field can itself be dispersive -- for instance, a Drude permittivity paired with a Lorentz-resonant permeability. There is no cross-coupling between the electric and magnetic responses in this model.

### 4.2 Bianisotropic Constitutive Form

The most general linear medium *does* couple the electric and magnetic responses. Writing the constitutive relation on the normalized field pair $[\mathbf{E};\ \eta_0\mathbf{H}]$ (with $\eta_0$ the free-space impedance), the response is a single $6\times6$ matrix

```math
\begin{pmatrix} \mathbf{D}/\varepsilon_0 \\ \eta_0\,\mathbf{B}/\mu_0 \end{pmatrix}
= \mathbf{C}_6 \begin{pmatrix} \mathbf{E} \\ \eta_0\mathbf{H} \end{pmatrix},
\qquad
\mathbf{C}_6 = \begin{pmatrix} \boldsymbol{\varepsilon}_r & \boldsymbol{\xi} \\ \boldsymbol{\zeta} & \boldsymbol{\mu}_r \end{pmatrix}.
```

The two diagonal $3\times3$ blocks are the relative permittivity (rows/cols 1-3) and relative permeability (rows/cols 4-6). The off-diagonal blocks $\boldsymbol{\xi}, \boldsymbol{\zeta}$ are the magnetoelectric coupling; for an *uncoupled* medium they are zero and $\mathbf{C}_6 = \operatorname{diag}(\boldsymbol{\varepsilon}_r, \boldsymbol{\mu}_r)$.

`BianisotropicMaterial3D(C6)` stores the normalized $6\times6$ as an `SMatrix{6,6}`. Passivity uses the same anti-Hermitian rule lifted to six dimensions: $(\mathbf{C}_6 - \mathbf{C}_6^\dagger)/(2i)$ must be negative semidefinite. The volume DDA path converts this normalized constitutive tensor into the solver's $[\mathbf{E};\mathbf{H}] \to [\mathbf{q};\mathbf{m}]$ polarizability convention.

---

## 5. Coupling to the Volume Solver

### 5.1 Evaluate First, Then Solve

The volume solver does **not** accept the material-model objects directly. `solve_dda_3d` / `assemble_dda_3d` accept `eps_r` as one of:

- a `Number` (isotropic, same for every voxel),
- a $3\times3$ `AbstractMatrix` (full tensor, same for every voxel),
- a 3-tuple of `Number`s (diagonal tensor),
- or a per-voxel collection of any of those.

The workflow is therefore: **evaluate the model at your operating frequency, then pass the resulting array.** A scalar model evaluates to `ComplexF64`; a diagonal model to `SVector{3}`; a tensor model to `SMatrix{3,3}`. Each of those is exactly a shape `solve_dda_3d` understands.

```julia
eps_now = material_epsr_3d(drude, freq_hz)   # ComplexF64 at this frequency
res = solve_dda_3d(grid, k0, eps_now, E_inc) # pass the evaluated value
```

The coercion logic lives in `_coerce_epsr_material_3d` (`DDA3D.jl`).

### 5.2 The Clausius-Mossotti Polarizability

Internally each voxel of relative permittivity $\varepsilon_r$ and volume $V$ is assigned the normalized electric polarizability

```math
\alpha_0 = 3V\,\frac{\varepsilon_r - 1}{\varepsilon_r + 2},
```

the Clausius-Mossotti form, which is the exact electrostatic polarizability of a sphere of the same volume. (The tensor version replaces the scalar fraction with $3V(\boldsymbol{\varepsilon}_r - \mathbf{I})(\boldsymbol{\varepsilon}_r + 2\mathbf{I})^{-1}$.) An optional radiation-reaction correction consistent with $e^{+i\omega t}$ is available:

```math
\alpha = \frac{\alpha_0}{1 + i\,k^3\alpha_0/(6\pi)}.
```

This polarizability is what the per-voxel solve returns in `res.alpha`, and it is the bridge from the constitutive description to the dipole physics.

---

## 6. Worked Example

The following script is fully self-contained. It builds every model type, evaluates the dispersive laws (and sweeps Drude passivity), and then feeds *evaluated* permittivities -- scalar and tensor -- into `solve_dda_3d` for a single voxel and a tiny $2\times2\times2$ grid. Run it with `julia --project=/path/to/DiffMoM.jl worked_example.jl`; it finishes in a few seconds and prints `ALL MATERIAL-MODEL CHECKS PASSED`.

```julia
using DiffMoM
using LinearAlgebra

println("== 1. Static models + evaluators ==")

# Isotropic permittivity. Passive loss requires imag(eps_r) <= 0.
iso = IsotropicMaterial3D(2.5 - 0.1im)
@assert material_epsr_3d(iso, 1.0e9) == 2.5 - 0.1im
println("iso  eps_r(1 GHz) = ", material_epsr_3d(iso, 1.0e9))

# Diagonal anisotropic permittivity (principal-axis 3-vector).
diag = DiagonalAnisotropicMaterial3D((2.0 - 0.1im, 3.0 - 0.2im, 4.0 + 0.0im))
println("diag eps_r        = ", collect(material_epsr_3d(diag, 3.0)))

# Full 3x3 tensor permittivity. Passive => (eps - eps')/(2im) negative semidef.
tensor = TensorAnisotropicMaterial3D(ComplexF64[
    2.0-0.1im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 3.0-0.2im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 4.0-0.3im
])
eps_t = material_epsr_3d(tensor, 3.0)
loss = (eps_t - adjoint(eps_t)) / (2im)
@assert maximum(eigvals(Hermitian(Matrix(loss)))) <= 100 * eps(Float64)
println("tensor size       = ", size(eps_t), "  passive loss check OK")

# Magnetodielectric medium: pairs a permittivity model with a permeability model.
mu = IsotropicPermeability3D(1.2 - 0.05im)
magnetic = MagneticMaterial3D(iso, mu)
@assert material_epsr_3d(magnetic, 1.0e9) == material_epsr_3d(iso, 1.0e9)
@assert material_mur_3d(magnetic, 1.0e9) == 1.2 - 0.05im
println("magnetic eps/mu   = ", material_epsr_3d(magnetic, 1.0e9), " / ",
        material_mur_3d(magnetic, 1.0e9))

# Bianisotropic: normalized 6x6 acting on [E; eta0*H]; diagonal blocks are
# relative permittivity (1:3) and permeability (4:6); off-diagonal = coupling.
C6 = Matrix{ComplexF64}(I, 6, 6)
C6[1, 1] = 2.0 - 0.01im
C6[4, 4] = 1.3 - 0.02im
C6[1, 5] = 0.05 + 0.0im
C6[5, 1] = 0.05 + 0.0im
bianiso = BianisotropicMaterial3D(C6)
@assert material_bianisotropic_matrix_3d(bianiso, 2.0) == bianiso.C6
println("bianiso C6[1,1]   = ", material_bianisotropic_matrix_3d(bianiso, 2.0)[1, 1])

println("\n== 2. Dispersive models (frequency in Hz) ==")

# Drude: eps = eps_inf - omega_p^2 / (omega^2 - i*gamma*omega),
# omega = 2pi*f, omega_p = 2pi*plasma_freq_hz, gamma = 2pi*gamma_hz.
drude = DrudePermittivity3D(1.0, 1.0e15, 1.0e13)
eps_drude = material_epsr_3d(drude, 2.0e14)          # via the model
eps_drude2 = drude_epsr_3d(2.0e14; eps_inf=1.0, plasma_freq_hz=1.0e15, gamma_hz=1.0e13)
@assert eps_drude == eps_drude2                       # model delegates to standalone fn
@assert imag(eps_drude) <= 0                          # passive
println("Drude   eps(2e14) = ", eps_drude)

# Lorentz: eps = eps_inf + strength*omega_0^2/(omega_0^2 - omega^2 + i*gamma*omega).
lorentz = LorentzPermittivity3D(1.0, 0.5, 2.0e14, 1.0e13)
eps_lor = material_epsr_3d(lorentz, 1.0e14)
@assert imag(eps_lor) <= 0
println("Lorentz eps(1e14) = ", eps_lor)

# Debye: eps = eps_inf + (eps_static - eps_inf)/(1 + i*omega*tau).
debye = DebyePermittivity3D(4.0, 2.0, 1.0e-10)
eps_deb = material_epsr_3d(debye, 1.0e9)
@assert imag(eps_deb) <= 0
println("Debye   eps(1e9)  = ", eps_deb)

# Passivity sweep for Drude across a band: imag(eps) <= 0 everywhere.
fs = range(5.0e13, 9.0e14; length=25)
@assert all(f -> imag(drude_epsr_3d(f; eps_inf=1.0, plasma_freq_hz=1.0e15,
                                    gamma_hz=1.0e13)) <= 0, fs)
println("Drude passivity sweep over 25 freqs: imag(eps) <= 0 everywhere")

# passive=false bypasses the check (e.g. gain media with imag(eps) > 0).
gain = IsotropicMaterial3D(2.0 + 0.1im; passive=false)
println("gain (passive=false) eps_r = ", gain.eps_r)

println("\n== 3. End-to-end: feed evaluated permittivity into solve_dda_3d ==")

k0 = 2π   # free-space wavenumber, length units consistent with the grid

# (a) Isotropic material -> scalar eps -> solver.
# Evaluate the model at the operating frequency, then pass the scalar to the
# solver. (solve_dda_3d's eps_r argument takes Number / 3x3 matrix / per-voxel
# collections, NOT the model objects directly -- evaluate first.)
grid1 = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 1, 1, 1)
eps_iso = material_epsr_3d(IsotropicMaterial3D(2.5 + 0.0im), k0)  # static scalar
E_inc1 = planewave_dda_3d(grid1, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
res1 = solve_dda_3d(grid1, k0, eps_iso, E_inc1)
println("isotropic solve: alpha[1] = ", res1.alpha[1])
@assert res1.alpha[1] ≈ clausius_mossotti_polarizability(eps_iso, grid1.volumes[1])

# (b) Anisotropic tensor material -> 3x3 SMatrix from material_epsr_3d -> solver.
# This tensor has asymmetric off-diagonals whose anti-Hermitian part is not
# negative semidefinite, so the passivity check would reject it; pass
# passive=false to model it anyway.
aniso = TensorAnisotropicMaterial3D(ComplexF64[
    2.5  0.12 0.0
    0.04 1.8  0.0
    0.0  0.0  1.0
]; passive=false)
eps_mat = material_epsr_3d(aniso, k0)   # SMatrix{3,3,ComplexF64,9}
E_inc2 = [CVec3(1.0 + 0im, 0.25 + 0im, 0.0 + 0im)]
res2 = solve_dda_3d(grid1, k0, eps_mat, E_inc2)
alpha_expected = clausius_mossotti_polarizability(eps_mat, grid1.volumes[1])
@assert res2.alpha[1] ≈ alpha_expected
println("anisotropic solve: alpha[1] (3x3) trace = ", tr(res2.alpha[1]))

# (c) Tiny multi-voxel solve with the same evaluated scalar permittivity.
grid2 = VoxelGrid3D((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 2, 2, 2)
E_inc3 = planewave_dda_3d(grid2, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
res3 = solve_dda_3d(grid2, k0, eps_iso, E_inc3)
F = farfield_dda_3d(res3, Vec3(0.0, 1.0, 0.0))
println("8-voxel solve far-field |F| = ", round(norm(F); sigdigits=6))

println("\nALL MATERIAL-MODEL CHECKS PASSED")
```

### 6.1 What the Example Demonstrates

- **Evaluation shapes:** a scalar model returns `ComplexF64`, a diagonal model an `SVector{3}`, a tensor model an `SMatrix{3,3}` -- the three shapes the solver consumes.
- **Tensor passivity:** the diagonal-lossy tensor passes the anti-Hermitian eigenvalue check, while the asymmetric tensor in part (b) must use `passive=false`.
- **Dispersion:** the Drude metal at $2\times10^{14}$ Hz returns a negative real, negative imaginary $\varepsilon_r \approx -23.94 - 1.25i$, and the passivity sweep confirms $\operatorname{Im}\varepsilon_r \le 0$ across the band.
- **Solver coupling:** the evaluated permittivities flow into `solve_dda_3d`, and the recovered `res.alpha[1]` matches `clausius_mossotti_polarizability` for both the scalar and the tensor case.

---

## 7. Validation

The behavior in this chapter is covered by the repository's test suite and a standalone ground-truth script:

- **`test/test_material_models3d.jl`** -- the dedicated `"3D material model helpers"` testset (20 assertions). It checks static iso/diagonal/tensor evaluation; the tensor passivity eigenvalue criterion `maximum(eigvals(Hermitian((eps - eps')/(2im)))) <= 100*eps(Float64)`; permeability models; `MagneticMaterial3D` delegation of $\varepsilon$ to `eps_model` and $\mu$ to `mu_model`; the bianisotropic $6\times6$ round-trip; passivity of the Drude/Lorentz/Debye evaluators at sample frequencies; and the negative cases (a non-passive scalar throws, `passive=false` bypasses the check, wrong shapes throw, `Inf` frequency throws).
- **`test/test_mom3d.jl`** -- `"Single-voxel Rayleigh dipole far field"` and `"Anisotropic tensor polarizability"` validate that a scalar permittivity and a $3\times3$ tensor flow through `solve_dda_3d` to the expected Clausius-Mossotti polarizability (the same `eps_r` coercion path used in the Worked Example).
- **`test/test_mom3d_em.jl`** -- `"Bianisotropic constitutive closure"` proves end-to-end consumption of `BianisotropicMaterial3D` by the EM volume solver: a diagonal $\mathbf{C}_6$ reproduces the scalar `(epsr, mur)` polarizabilities to machine precision, and adding off-diagonal coupling produces nonzero magnetoelectric polarizability blocks.

Run the material-model suite directly with:

```julia
julia --project=. test/test_material_models3d.jl
```

---

## 8. When to Use / Limitations

**Use these models when:**

- You need a frequency-dependent permittivity (plasma metals via Drude, resonant dielectrics via Lorentz, polar liquids via Debye).
- The scatterer is anisotropic or magnetodielectric, or has magnetoelectric coupling (bianisotropic).
- You want passivity enforced automatically as a guard against sign-convention or data-entry mistakes.

**Limitations to keep in mind:**

- The models are *pure constitutive descriptions*. They do not solve anything; you must evaluate them and pass the result to the volume solver.
- Dispersion is single-pole. Multi-pole or tabulated $\varepsilon(\omega)$ is not provided as a built-in model -- supply the evaluated `ComplexF64` directly at each frequency, or construct a tensor per frequency.
- The Clausius-Mossotti polarizability is a same-volume-sphere electrostatic approximation; it is most accurate for sub-wavelength voxels. Enable the radiation-reaction correction for larger voxels.
- Passivity checks are construction-time (and evaluation-time for the dispersive functions) only; they do not re-validate when you mutate a stored field.

---

## 9. Code Mapping

| Concept | Exported symbol | Source file |
|---------|-----------------|-------------|
| Isotropic permittivity | `IsotropicMaterial3D` | `src/mom3d/MaterialModels3D.jl` |
| Diagonal anisotropic permittivity | `DiagonalAnisotropicMaterial3D` | `src/mom3d/MaterialModels3D.jl` |
| Full tensor permittivity | `TensorAnisotropicMaterial3D` | `src/mom3d/MaterialModels3D.jl` |
| Isotropic / diagonal / tensor permeability | `IsotropicPermeability3D`, `DiagonalPermeability3D`, `TensorPermeability3D` | `src/mom3d/MaterialModels3D.jl` |
| Drude / Lorentz / Debye models | `DrudePermittivity3D`, `LorentzPermittivity3D`, `DebyePermittivity3D` | `src/mom3d/MaterialModels3D.jl` |
| Magnetodielectric medium | `MagneticMaterial3D` | `src/mom3d/MaterialModels3D.jl` |
| Bianisotropic medium | `BianisotropicMaterial3D` | `src/mom3d/MaterialModels3D.jl` |
| Permittivity / permeability evaluators | `material_epsr_3d`, `material_mur_3d` | `src/mom3d/MaterialModels3D.jl` |
| Bianisotropic matrix evaluator | `material_bianisotropic_matrix_3d` | `src/mom3d/MaterialModels3D.jl` |
| Standalone dispersive evaluators | `drude_epsr_3d`, `lorentz_epsr_3d`, `debye_epsr_3d` | `src/mom3d/MaterialModels3D.jl` |
| `eps_r` coercion into the solver | `_coerce_epsr_material_3d` (internal) | `src/mom3d/DDA3D.jl` |
| Per-voxel polarizability | `clausius_mossotti_polarizability` | `src/mom3d/DDA3D.jl` |
| Volume solve / assembly | `solve_dda_3d`, `assemble_dda_3d` | `src/mom3d/DDA3D.jl` |
| Voxel grid / vector types | `VoxelGrid3D`, `Vec3`, `CVec3` | `src/mom3d/Types3D.jl`, `src/Types.jl` |

---

## 10. Exercises

### 10.1 Conceptual

1. **Sign convention.** Under $e^{-i\omega t}$, what sign does $\operatorname{Im}\varepsilon_r$ take for a passive lossy dielectric? Explain why constructing `IsotropicMaterial3D(2.0 + 0.1im)` throws under the package's $e^{+i\omega t}$ convention but the conjugate value `2.0 - 0.1im` succeeds.

2. **Tensor passivity.** Show by hand that the tensor $\begin{psmallmatrix} 2.5 & 0.12 & 0 \\ 0.04 & 1.8 & 0 \\ 0 & 0 & 1 \end{psmallmatrix}$ has a nonzero anti-Hermitian part even though every entry is real, and argue why the largest eigenvalue of $(\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}^\dagger)/(2i)$ is positive. What is the smallest symmetric perturbation that would make it pass the default check?

3. **Units.** A student passes a plasma frequency in rad/s instead of Hz to `DrudePermittivity3D`. By what factor is the resulting plasma frequency wrong, and in which direction does the $\operatorname{Re}\varepsilon_r = 0$ crossover shift?

### 10.2 Numerical

4. **Drude crossover.** Sweep `drude_epsr_3d` from $5\times10^{13}$ to $1.2\times10^{15}$ Hz with `plasma_freq_hz=1.0e15`, `gamma_hz=1.0e13` and find the frequency where $\operatorname{Re}\varepsilon_r$ crosses zero. Compare to the lossless estimate $f \approx f_p/\sqrt{\varepsilon_\infty}$.

5. **Lorentz resonance.** For `LorentzPermittivity3D(1.0, 0.5, 2.0e14, 1.0e13)`, plot $\operatorname{Re}\varepsilon_r$ and $-\operatorname{Im}\varepsilon_r$ across $1\times10^{14}$ to $3\times10^{14}$ Hz. Confirm the absorption peak ($-\operatorname{Im}\varepsilon_r$ maximum) sits near $f_0 = 2\times10^{14}$ Hz and that anomalous dispersion appears in the real part.

6. **Dispersive solve.** Evaluate a Drude permittivity at three frequencies, pass each scalar to `solve_dda_3d` on the single-voxel grid from the Worked Example, and tabulate how `res.alpha[1]` varies. Explain the trend in terms of $\varepsilon_r(\omega)$.

### 10.3 Advanced

7. **Per-voxel materials.** Build an $\varepsilon_r$ collection where half the voxels of a $2\times2\times2$ grid are a Drude metal and half are a lossless dielectric (a length-8 vector of evaluated scalars), pass it to `solve_dda_3d`, and inspect the resulting per-voxel polarizabilities.

8. **Magnetoelectric coupling.** Starting from the diagonal $\mathbf{C}_6$ in the Worked Example, add increasing off-diagonal coupling `C6[1,5] = C6[5,1] = c` for `c` in `[0.0, 0.02, 0.05]`. For each, check whether the default passivity test still passes, and (mirroring `test/test_mom3d_em.jl`) confirm that nonzero coupling produces nonzero magnetoelectric polarizability blocks in the EM solver.

---

## 11. Further Reading

1. **Constitutive relations and dispersion:**
   - Jackson, J. D. (1998). *Classical Electrodynamics*, 3rd ed. Wiley. Chapter 7 derives the Drude and Lorentz models and the Kramers-Kronig (passivity) constraints.
   - Born, M. & Wolf, E. (1999). *Principles of Optics*, 7th ed. Cambridge. Dispersion and absorption in dielectrics.

2. **Bianisotropic and complex media:**
   - Lindell, I. V., Sihvola, A. H., Tretyakov, S. A., & Viitanen, A. J. (1994). *Electromagnetic Waves in Chiral and Bi-isotropic Media*. Artech House.

3. **Discrete-dipole approximation:**
   - Draine, B. T. & Flatau, P. J. (1994). "Discrete-dipole approximation for scattering calculations." *J. Opt. Soc. Am. A*, 11(4), 1491-1499. Origin of the Clausius-Mossotti polarizability and the radiative-reaction correction.

4. **DiffMoM.jl source and API:** `src/mom3d/MaterialModels3D.jl` for the model implementations; the [3D Material Models API page](../api/material-models-3d.md) for exact signatures; the [3D Volume DDA API page](../api/dda-volume-3d.md) for the solver that consumes these models.

---

*Related: the [3D Material Models API page](../api/material-models-3d.md) lists every constructor and evaluator signature, and the [3D Volume DDA API page](../api/dda-volume-3d.md) documents `solve_dda_3d`, `clausius_mossotti_polarizability`, and the rest of the volume-scattering path that these models feed.*
