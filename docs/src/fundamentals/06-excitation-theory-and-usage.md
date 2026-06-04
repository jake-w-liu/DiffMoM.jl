# Excitation Theory and Implementation

## Purpose

Excitation modeling forms the critical link between the **known incident field** and the **right‑hand side (RHS)** of the EFIE linear system. This chapter provides a comprehensive physics‑to‑code explanation of all excitation models available in `DiffMoM.jl`, covering their mathematical foundations, implementation details, practical use cases, and validation pathways.

A common point of confusion in EFIE‑MoM simulations is distinguishing between solving for the **source radiation** itself versus computing the **scatterer response** to a known incident field. In this package, excitation models are **incident‑field generators**: they construct a known electric field $\mathbf{E}^{\mathrm{inc}}(\mathbf{r})$ on the scattering surface, which is then projected onto the RWG basis to form the RHS vector $\mathbf{v}$. The models are **not** standalone source‑radiation solvers unless you explicitly provide such a field from an external simulation.

---

## Learning Goals

After studying this chapter, you should be able to:

1. **Derive** the exact mathematical relationship between the incident field and the EFIE RHS vector.
2. **Explain** the physical assumptions and limitations of each excitation type: plane wave, dipole, loop, delta‑gap, port, imported field, and pattern feed.
3. **Choose** the appropriate excitation model based on your knowledge of the incident field and the simulation scenario.
4. **Implement** custom incident fields using the imported‑field interface while respecting quadrature requirements and vector‑field conventions.
5. **Interpret** the package’s validation gates (amplitude, polarization, phase consistency) as physical checks rather than arbitrary software tests.
6. **Diagnose** and fix common excitation‑related errors, such as time‑convention mismatches or insufficient quadrature resolution.

---

## 1. Mathematical Foundation: From Incident Field to RHS Vector

### 1.1 Boundary Condition and EFIE Formulation

For a perfect electric conductor (PEC), the total tangential electric field on the surface $\Gamma$ must vanish:

```math
\mathbf{E}^{\mathrm{tot}}_t(\mathbf{r}) = \mathbf{E}^{\mathrm{inc}}_t(\mathbf{r}) + \mathbf{E}^{\mathrm{sca}}_t(\mathbf{r}) = 0, \quad \mathbf{r} \in \Gamma,
```

where $\mathbf{E}^{\mathrm{inc}}$ is the **known incident field** and $\mathbf{E}^{\mathrm{sca}}$ is the **scattered field** produced by the induced surface current $\mathbf{J}$. The EFIE operator $\mathcal{T}$ maps the surface current to the scattered tangential field:

```math
\mathcal{T}[\mathbf{J}](\mathbf{r}) = \mathbf{E}^{\mathrm{sca}}_t(\mathbf{r}).
```

The boundary condition therefore becomes

```math
\mathcal{T}[\mathbf{J}](\mathbf{r}) = -\mathbf{E}^{\mathrm{inc}}_t(\mathbf{r}), \quad \mathbf{r} \in \Gamma.
```

### 1.2 Galerkin Discretization

Expand the unknown current in RWG basis functions $\{\mathbf{f}_n\}_{n=1}^N$:

```math
\mathbf{J}(\mathbf{r}) \approx \sum_{n=1}^N I_n \mathbf{f}_n(\mathbf{r}).
```

Apply Galerkin testing with the same basis functions:

```math
\langle \mathbf{f}_m, \mathcal{T}[\mathbf{J}] \rangle = -\langle \mathbf{f}_m, \mathbf{E}^{\mathrm{inc}} \rangle, \quad m = 1,\dots,N,
```

where the inner product is defined as

```math
\langle \mathbf{f}, \mathbf{g} \rangle = \int_\Gamma \mathbf{f}(\mathbf{r}) \cdot \mathbf{g}(\mathbf{r}) \, dS.
```

Substituting the current expansion yields the linear system

```math
\sum_{n=1}^N Z_{mn} I_n = v_m,
```

with the impedance matrix entries

```math
Z_{mn} = \langle \mathbf{f}_m, \mathcal{T}[\mathbf{f}_n] \rangle,
```

and the **excitation vector** (RHS) entries

```math
v_m = -\langle \mathbf{f}_m, \mathbf{E}^{\mathrm{inc}} \rangle.
```

Thus, **excitation enters the EFIE solely through the projection of the incident field onto the testing functions**.

### 1.3 Discrete Quadrature Implementation

For each RWG basis function $\mathbf{f}_m$, which is supported on two triangles $T_m^+$ and $T_m^-$, the discrete projection becomes

```math
v_m = -\sum_{T \in \{T_m^+, T_m^-\}} \int_T \mathbf{f}_m(\mathbf{r}) \cdot \mathbf{E}^{\mathrm{inc}}(\mathbf{r}) \, dS.
```

Using a reference‑triangle quadrature rule with points $\{\mathbf{r}_{T,q}\}$ and weights $\{w_q\}$,

```math
v_m \approx -\sum_{T} \sum_{q=1}^{N_q} w_q \, \mathbf{f}_m(\mathbf{r}_{T,q}) \cdot \mathbf{E}^{\mathrm{inc}}(\mathbf{r}_{T,q}) \, (2A_T),
```

where $A_T$ is the area of triangle $T$ and the factor $2A_T$ arises from the affine mapping from the reference triangle to the physical triangle. This quadrature formula is implemented exactly in the function `assemble_excitation` for all excitation types.

### 1.4 Time Convention and Field Phasors

The package uses the **$e^{+i\omega t}$ time convention**. Under this convention, a time‑harmonic field is expressed as

```math
\widetilde{\mathbf{E}}(\mathbf{r}, t) = \Re\{\mathbf{E}(\mathbf{r}) e^{+i\omega t}\},
```

and the free‑space Green’s function carries a phase factor $e^{-ikR}$ with $R = |\mathbf{r}-\mathbf{r}'|$ and $k = \omega\sqrt{\mu_0\epsilon_0}$. All incident‑field expressions in the code (plane wave, dipole, loop, pattern feed) are derived with this convention. When importing field data from external tools that use the opposite convention ($e^{-i\omega t}$), the pattern‑feed interface provides a `convention` keyword to automatically conjugate the coefficients.

### 1.5 Physical Units and Scaling

Throughout the excitation system, SI units are employed:
- Positions: meters (m)
- Electric field: volts per meter (V/m)
- Wavenumber $k$: radians per meter (rad/m)
- Frequencies: hertz (Hz)
- Dipole moments: electric dipole $\mathbf{p}$ in coulomb‑meters (C·m), magnetic dipole $\mathbf{m}$ in ampere‑square‑meters (A·m²)

The RHS vector $v_m$ has units of volts (V), consistent with the impedance matrix $Z_{mn}$ having units of ohms (Ω).

---

## 2. Plane‑Wave Excitation (`PlaneWaveExcitation`)

### 2.1 Mathematical Model

A uniform plane wave is characterized by a propagation direction $\hat{\mathbf{k}}$ (unit vector), a complex amplitude $E_0$ (V/m), and a polarization vector $\mathbf{p}$ (unit vector orthogonal to $\hat{\mathbf{k}}$). The incident electric field is

```math
\mathbf{E}^{\mathrm{inc}}(\mathbf{r}) = \mathbf{p} \, E_0 \, e^{-i k \hat{\mathbf{k}} \cdot \mathbf{r}},
```

where $k = \omega/c_0$ is the wavenumber. In the code, the wave vector is stored as $\mathbf{k}_{\mathrm{vec}} = k \hat{\mathbf{k}}$.

### 2.2 Implementation

The struct `PlaneWaveExcitation` holds the three fields `k_vec`, `E0`, and `pol`. The assembly function `assemble_plane_wave` evaluates the field at each quadrature point using the helper `plane_wave_field` and performs the projection described in Section 1.3.

**Backward compatibility**: The legacy function `assemble_v_plane_wave` still exists but simply creates a `PlaneWaveExcitation` and calls the unified `assemble_excitation` dispatcher.

### 2.3 Use Cases and Limitations

- **RCS calculations**: Plane‑wave excitation is the standard illuminator for radar cross‑section (RCS) simulations.
- **Cross‑validation**: Comparison with analytical solutions (Mie series for spheres, physical‑optics approximations) typically uses plane‑wave incidence.
- **Canonical scattering**: Many benchmark problems (e.g., the NASA almond, dihedral corners) are defined for plane‑wave illumination.

**Limitations**:
- A true plane wave is an idealization that assumes an infinite, uniform wavefront. For electrically large scatterers, edge‑diffraction effects may differ from more realistic (finite‑beam) illuminations.
- The model does not account for wave‑front curvature, which can be important for near‑field sources.

### 2.4 Example Usage

```julia
using DiffMoM

# Define plane‑wave parameters
freq = 1.0e9                    # 1 GHz
λ = 299792458.0 / freq          # wavelength
k = 2π / λ                      # wavenumber
k_vec = Vec3(0.0, 0.0, -k)     # propagating in ‑z direction
E0 = 1.0                        # amplitude 1 V/m
pol = Vec3(1.0, 0.0, 0.0)      # x‑polarized

# Create excitation object
pw = make_plane_wave(k_vec, E0, pol)

# Assemble RHS vector for a given mesh and RWG basis
mesh = make_rect_plate(0.1, 0.1, 10, 10)
rwg = build_rwg(mesh)
v = assemble_excitation(mesh, rwg, pw; quad_order=3)
```

---

## 3. Dipole Excitation (`DipoleExcitation`)

### 3.1 Mathematical Model

An electric or magnetic dipole is a canonical localized source whose closed‑form field expression is known everywhere in free space. For an **electric dipole** with moment $\mathbf{p}$ (C·m) located at $\mathbf{r}_0$, the electric field at observation point $\mathbf{r}$ is

```math
\mathbf{E}_{\mathrm{elec}}(\mathbf{r}) = \frac{e^{-ikR}}{4\pi\epsilon_0 R}
\Bigl[ k^2 \hat{\mathbf{R}} \times (\mathbf{p} \times \hat{\mathbf{R}})
+ \Bigl(\frac{1}{R^2} - i\frac{k}{R}\Bigr)
\bigl(3\hat{\mathbf{R}}(\hat{\mathbf{R}}\cdot\mathbf{p}) - \mathbf{p}\bigr) \Bigr],
```

where $R = |\mathbf{r}-\mathbf{r}_0|$ and $\hat{\mathbf{R}} = (\mathbf{r}-\mathbf{r}_0)/R$. The first term dominates in the far field ($kR \gg 1$), while the second term captures the near‑field ($kR \lesssim 1$) contributions.

For a **magnetic dipole** with moment $\mathbf{m}$ (A·m²),

```math
\mathbf{E}_{\mathrm{mag}}(\mathbf{r}) = \frac{\eta_0}{4\pi}
\Bigl(\frac{k}{R^2} + i\frac{k^2}{R}\Bigr) e^{-ikR} (\mathbf{m} \times \hat{\mathbf{R}}),
```

where $\eta_0 = \sqrt{\mu_0/\epsilon_0} \approx 376.73\,\Omega$ is the free‑space impedance.

### 3.2 Implementation

The struct `DipoleExcitation` stores the dipole position `position`, moment `moment` (as a complex 3‑vector), orientation metadata `orientation`, type `type` (`:electric` or `:magnetic`), and frequency `frequency`. The frequency is required to compute the wavenumber $k$.

The field evaluation is performed by `dipole_incident_field`, which implements the above formulas with careful handling of the singular case $R \to 0$ (returns zero field). The assembly then proceeds via the standard quadrature projection.

### 3.3 Use Cases

- **Localized source modeling**: Dipoles are physically realistic models for small antennas (short wires, small loops).
- **Near‑field/far‑field transition studies**: The exact expression captures both regimes, allowing verification of numerical far‑field extraction routines.
- **Polarization basis checks**: An electric dipole oriented along $x$ produces predominantly $E_\theta$ polarization in the far field, while a magnetic dipole along $z$ produces $E_\phi$; this difference is used in validation tests.
- **Equivalent source approximation**: A complex current distribution can sometimes be approximated by a dipole moment (e.g., via moment integration) for quick feasibility studies.

### 3.4 Example: Electric Dipole

```julia
using DiffMoM

freq = 2.0e9                    # 2 GHz
position = Vec3(0.0, 0.0, 0.05) # 5 cm above origin
moment = CVec3(1e-9, 0.0, 0.0)  # 1 nC·m along x
orientation = Vec3(1.0, 0.0, 0.0)
dip = make_dipole(position, moment, orientation, :electric, freq)

mesh = make_rect_plate(0.1, 0.1, 10, 10)
rwg = build_rwg(mesh)
v = assemble_excitation(mesh, rwg, dip; quad_order=3)
```

---

## 4. Loop Excitation (`LoopExcitation`)

### 4.1 Mathematical Model

A small circular loop of radius $a$ carrying a uniform current $I$ is equivalent to a **magnetic dipole** oriented normal to the loop plane. The equivalent magnetic dipole moment is

```math
\mathbf{m} = I \, \pi a^2 \, \hat{\mathbf{n}},
```

where $\hat{\mathbf{n}}$ is the unit normal vector. The electric field radiated by the loop is then computed using the magnetic‑dipole formula from Section 3.1.

### 4.2 Implementation

The struct `LoopExcitation` contains the loop center `center`, normal `normal`, radius `radius`, current `current`, and frequency `frequency`. The field evaluation function `loop_incident_field` constructs an equivalent `DipoleExcitation` of type `:magnetic` and calls `dipole_incident_field`.

**Validation**: The package includes a test that verifies the loop field matches the equivalent magnetic‑dipole field to within machine precision.

### 4.3 Use Cases

- **Small‑loop antennas**: Electrically small loops ($ka \ll 1$) are accurately modeled by this approximation.
- **Magnetic‑source prototyping**: Useful for scenarios where a magnetic source is more natural than an electric dipole.
- **Polarization comparison**: Loops and electric dipoles produce orthogonal polarizations, enabling checks of cross‑polarization isolation.

### 4.4 Example

```julia
using DiffMoM

freq = 500e6                    # 500 MHz
center = Vec3(0.0, 0.0, 0.1)    # 10 cm above origin
normal = Vec3(0.0, 0.0, 1.0)    # loop in xy‑plane
radius = 0.005                  # 5 mm radius
current = 1.0 + 0.0im           # 1 A
loop = make_loop(center, normal, radius, current, freq)

mesh = make_rect_plate(0.1, 0.1, 10, 10)
rwg = build_rwg(mesh)
v = assemble_excitation(mesh, rwg, loop; quad_order=3)
```

---

## 5. Delta‑Gap and Port Excitations

### 5.1 Delta‑Gap Model (`DeltaGapExcitation`)

A delta‑gap is a **lumped feed approximation** that applies a voltage $V$ across a **single RWG edge**, assumed to be separated by a small physical gap of length $g$. The excitation vector is

```math
v_m \approx \frac{V}{g} \, \delta_{m, m_{\mathrm{gap}}},
```

where $\delta_{m, m_{\mathrm{gap}}}$ is 1 if $m$ is the gap edge and 0 otherwise. This corresponds to an incident field that is non‑zero only within the infinitesimal gap region, approximated as a uniform electric field $V/g$ across the gap.

#### Implementation

The struct `DeltaGapExcitation` stores the edge index `edge`, voltage `voltage`, and gap length `gap_length`. The assembly function `assemble_delta_gap` creates a vector with a single non‑zero entry at the specified edge.

#### Use Cases

- **Simple feed modeling**: Quick prototyping of edge‑driven antennas.
- **Circuit‑like excitations**: When the feed is modeled as a voltage source in series with a gap.
- **Validation**: The delta‑gap model is analytically simple, making it useful for debugging the overall EFIE pipeline.

### 5.2 Port Model (`PortExcitation`)

A port generalizes the delta‑gap to **multiple RWG edges**, representing a distributed feed region. The current implementation applies a uniform voltage scaling across the listed edges, normalized by the edge length:

```math
v_m = \frac{V}{\ell_m} \quad \text{for } m \in \text{port\_edges},
```

and $v_m = 0$ otherwise. The struct also stores a port impedance `impedance` (Ω) for metadata, though this is not used in the EFIE assembly.

**Important**: This is **not** a full waveguide‑port model with modal field expansion. It is a lumped approximation suitable for preliminary design where the exact port fields are not critical.

#### Use Cases

- **Multi‑edge feeds**: Antennas fed across several edges (e.g., a microstrip patch).
- **Impedance‑matching studies**: The port impedance metadata can be used in external circuit simulations.
- **Optimization**: When the exact feed field is not the focus, a port excitation simplifies the problem.

### 5.3 Example: Delta‑Gap and Port

```julia
using DiffMoM

mesh = make_rect_plate(0.1, 0.1, 10, 10)
rwg = build_rwg(mesh)

# Delta‑gap on edge 5
gap = make_delta_gap(5, 1.0 + 0.0im, 0.001)  # 1 V across 1 mm gap
v_gap = assemble_excitation(mesh, rwg, gap)

# Port on edges 5, 6, 7
port = PortExcitation([5, 6, 7], 1.0 + 0.0im, 50.0)
v_port = assemble_excitation(mesh, rwg, port)
```

---

## 6. Imported Field Excitation (`ImportedExcitation`)

### 6.1 Purpose and Scope

The `ImportedExcitation` model allows you to provide an **arbitrary incident‑field function** $\mathbf{E}^{\mathrm{inc}}(\mathbf{r})$ (or an equivalent surface‑current density) that is then projected onto the RWG basis. This is the most flexible excitation type, but it must be used with a clear understanding of its scope:

- **It is an incident‑field generator**, not a standalone source‑radiation solver.
- The provided function must return the incident electric field **at the scatterer surface** (or an equivalent quantity).
- It does **not** solve for the radiation from arbitrary volumetric currents; you must already know the incident field from another source (analytic expression, external full‑wave simulation, measurement).

### 6.2 Two Semantic Variants

The constructor `ImportedExcitation` accepts a `kind` keyword with two options:

1. **`:electric_field`** (default): The function `source_func(r)` returns the incident electric field phasor $\mathbf{E}^{\mathrm{inc}}(\mathbf{r})$ directly. This is the most straightforward usage.

2. **`:surface_current_density`**: The function `source_func(r)` returns an equivalent surface current density $\mathbf{J}_s(\mathbf{r})$ (A/m). The incident field is then approximated locally as

```math
\mathbf{E}^{\mathrm{inc}}(\mathbf{r}) \approx \eta_{\mathrm{equiv}} \, \mathbf{J}_s(\mathbf{r}),
```

where $\eta_{\mathrm{equiv}}$ is a user‑supplied equivalent impedance (defaults to $\eta_0$). This is useful for aperture or sheet‑like approximations where the incident field is related to a known surface current.

### 6.3 Implementation Details

The struct `ImportedExcitation` stores the function `source_func`, the kind `kind`, the equivalent impedance `eta_equiv`, and a `min_quad_order` parameter. The assembly function `assemble_imported_excitation`:

1. Determines the effective quadrature order as `max(user_quad_order, min_quad_order)` and rounds up to the nearest supported rule (orders 1, 3, 4, or 7).
2. Evaluates `source_func` at each quadrature point.
3. Converts the return value to a complex 3‑vector (`CVec3`) using the helper `_to_cvec3`, which accepts `CVec3`, `Vec3`, `SVector`, tuples, or length‑3 vectors, and checks for finiteness.
4. Applies the $\eta_{\mathrm{equiv}}$ scaling if `kind = :surface_current_density`.
5. Projects the field onto the RWG basis via the standard quadrature formula.

### 6.4 Use Cases

- **Custom analytic fields**: Any incident field that can be written as a closed‑form expression.
- **Fields from external solvers**: Import near‑field data from a separate full‑wave simulation (FEM, FDTD) and interpolate it onto the MoM mesh.
- **Aperture‑field approximations**: When the incident field is known only on an aperture plane, the `:surface_current_density` kind can map it to an equivalent surface current.
- **Deterministic field prescriptions**: For specialized studies where the incident field must follow a specific spatial pattern.

### 6.5 Example: Imported Plane Wave

```julia
using DiffMoM

freq = 3.0e9
k = 2π * freq / 299792458.0

# Define an x‑polarized plane wave propagating in ‑z
E_imported(r) = CVec3(exp(-1im * k * r[3]), 0.0 + 0im, 0.0 + 0im)

# Create imported excitation with minimum quadrature order 3
exc = ImportedExcitation(E_imported; kind=:electric_field, min_quad_order=3)

mesh = make_rect_plate(0.1, 0.1, 10, 10)
rwg = build_rwg(mesh)
v = assemble_excitation(mesh, rwg, exc; quad_order=3)
```

### 6.6 Common Pitfalls and Fixes

1. **Return type mismatch**: Ensure your function returns a 3‑component numeric vector. The helper `_to_cvec3` will catch common mistakes and give a clear error message.
2. **Insufficient quadrature**: If the imported field varies rapidly over a triangle, increase `min_quad_order` (or the assembly `quad_order`) to capture the variation accurately.
3. **Time‑convention mismatch**: If the imported field comes from a tool using $e^{-i\omega t}$, you must **conjugate** the field values before passing them to `ImportedExcitation`. Alternatively, multiply the function by `conj` inside the wrapper.
4. **Physical consistency**: The imported field should satisfy Maxwell’s equations (or at least be a plausible incident field). The package does not check this; it is the user’s responsibility.

---

## 7. Pattern‑Feed Excitation (`PatternFeedExcitation`)

### 7.1 Mathematical Model

A pattern feed imports far‑field coefficients $F_\theta(\theta,\phi)$ and $F_\phi(\theta,\phi)$ on a spherical grid $(\theta_i, \phi_j)$. The incident electric field at a point $\mathbf{r}$ (relative to the phase center $\mathbf{r}_{\mathrm{pc}}$) is synthesized as

```math
\mathbf{E}^{\mathrm{inc}}(\mathbf{r}) = \frac{e^{-ikR}}{R}
\bigl[ F_\theta(\theta,\phi) \, \hat{\boldsymbol{\theta}}
     + F_\phi(\theta,\phi)   \, \hat{\boldsymbol{\phi}} \bigr],
```

where $R = |\mathbf{r} - \mathbf{r}_{\mathrm{pc}}|$, $(\theta,\phi)$ are the spherical angles of $\mathbf{r} - \mathbf{r}_{\mathrm{pc}}$, and $(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}})$ are the corresponding unit vectors. The coefficients are defined under the **$e^{+i\omega t}$ convention**; if the imported data use $e^{-i\omega t}$, the `convention` keyword automatically conjugates them.

### 7.2 Implementation

The struct `PatternFeedExcitation` stores:
- `theta`, `phi`: strictly increasing grids (radians).
- `Ftheta`, `Fphi`: complex matrices of size `(length(theta), length(phi))`.
- `frequency`: needed for wavenumber $k$.
- `phase_center`: origin of the spherical coordinate system.
- `convention`: `:exp_plus_iwt` (default) or `:exp_minus_iwt`.

Field evaluation in `pattern_feed_field`:
1. Computes $R$ and $(\theta,\phi)$ from $\mathbf{r}$.
2. Performs **bilinear interpolation** on the spherical grid:
   - $\theta$ uses non‑periodic bracketing (clamped to grid ends).
   - $\phi$ uses periodic bracketing (wraps around $2\pi$).
3. Retrieves $F_\theta$, $F_\phi$ at the interpolated angles.
4. Applies conjugation if `convention = :exp_minus_iwt`.
5. Constructs the field vector in Cartesian coordinates.

### 7.3 Grid Requirements

- $\theta \in [0, \pi]$ (radians), strictly increasing.
- $\phi$ should cover **one open period** (e.g., $[0, 2\pi-\Delta]$) without a duplicate endpoint, to avoid ambiguity in periodic interpolation.
- Both $F_\theta$ and $F_\phi$ must be provided as **complex‑valued** matrices (amplitude and phase). Power‑only (gain) patterns are insufficient.

### 7.4 Use Cases

- **Horn‑feed illumination**: Import measured or simulated horn patterns to illuminate reflectors or other scatterers.
- **Array‑feed synthesis**: Combine multiple pattern feeds to model array excitations.
- **Cross‑tool workflows**: Use patterns from specialized antenna‑design software (CST, HFSS, FEKO) within the MoM framework.
- **Validation**: Compare MoM‑computed patterns with analytical dipole/loop patterns via `make_analytic_dipole_pattern_feed`.

### 7.5 Example: Creating a Pattern Feed from Arrays

```julia
using DiffMoM

freq = 10.0e9
λ = 299792458.0 / freq
k = 2π / λ

# Create spherical grids
Nθ = 91  # 0° to 180° in 2° steps
Nϕ = 181 # 0° to 360° in 2° steps
θ = range(0.0, π, length=Nθ)
ϕ = range(0.0, 2π - 2π/Nϕ, length=Nϕ)  # open period

# Synthetic pattern: x‑polarized dipole‑like
Fθ = zeros(ComplexF64, Nθ, Nϕ)
Fϕ = zeros(ComplexF64, Nθ, Nϕ)
for i in 1:Nθ, j in 1:Nϕ
    Fθ[i,j] = cos(θ[i]) * exp(-1im * k * sin(θ[i])*cos(ϕ[j]))
    Fϕ[i,j] = 0.0 + 0im
end

pat = make_pattern_feed(θ, ϕ, Fθ, Fϕ, freq;
                        phase_center=Vec3(0,0,0),
                        angles_in_degrees=false,
                        convention=:exp_plus_iwt)

mesh = make_rect_plate(0.2, 0.2, 20, 20)
rwg = build_rwg(mesh)
v = assemble_excitation(mesh, rwg, pat; quad_order=4)
```

### 7.6 Helper: Analytic Dipole Pattern Feed

The function `make_analytic_dipole_pattern_feed` creates a `PatternFeedExcitation` from a `DipoleExcitation` object, evaluating the analytical far‑field coefficients on the supplied $(\theta,\phi)$ grid. This is useful for validation and for creating pattern feeds when you have a dipole source but want the pattern‑feed interface (e.g., for consistency with other pattern‑based workflows).

```julia
dip = make_dipole(Vec3(0,0,0), CVec3(1e-9,0,0), Vec3(1,0,0), :electric, 1e9)
θ = range(0.0, π, length=91)
ϕ = range(0.0, 2π-2π/180, length=180)
pat_dip = make_analytic_dipole_pattern_feed(dip, θ, ϕ; phase_center=dip.position)
```

---

## 8. Multi‑Excitation (`MultiExcitation`)

### 8.1 Linear Superposition Principle

Because the EFIE is linear in the incident field, the RHS vector for a weighted combination of excitations is the corresponding weighted sum of the individual RHS vectors:

```math
\mathbf{v} = \sum_{j=1}^{M} w_j \, \mathbf{v}^{(j)},
```

where $\mathbf{v}^{(j)}$ is the RHS from the $j$‑th excitation and $w_j$ is a complex weight. This enables scenarios with multiple simultaneous sources or with parameterized excitation families.

### 8.2 Implementation

The struct `MultiExcitation` holds a vector of `excitations` (any subtype of `AbstractExcitation`) and a vector of complex `weights`. The assembly function `assemble_multi` simply loops over the excitations, calls `assemble_excitation` for each, and accumulates the weighted results.

### 8.3 Use Cases

- **Multiple incident angles**: Synthesize a scenario with several plane waves arriving from different directions.
- **Weighted source combinations**: Model a feed network with amplitude/phase weighting across multiple ports or dipoles.
- **Parameter sweeps**: When solving many RHS vectors that are linear combinations of a few basis excitations, use `assemble_multiple_excitations` to obtain the matrix $\mathbf{V}$ directly (see Section 9).

### 8.4 Example

```julia
using DiffMoM

mesh = make_rect_plate(0.1, 0.1, 10, 10)
rwg = build_rwg(mesh)

# Create two plane waves with different propagation directions
k = 2π / 0.3  # wavelength 0.3 m
pw1 = make_plane_wave(Vec3(0,0,-k), 1.0, Vec3(1,0,0))
pw2 = make_plane_wave(Vec3(-k,0,0), 0.5, Vec3(0,0,1))

# Combine with weights 0.7 and 0.3
multi = make_multi_excitation([pw1, pw2], [0.7, 0.3])
v_multi = assemble_excitation(mesh, rwg, multi; quad_order=3)

# Verify linearity: should match manual combination
v1 = assemble_excitation(mesh, rwg, pw1; quad_order=3)
v2 = assemble_excitation(mesh, rwg, pw2; quad_order=3)
v_manual = 0.7*v1 + 0.3*v2
println("Linear combination error: ", norm(v_multi - v_manual) / norm(v_manual))
```

---

## 9. Multiple Excitations Matrix Assembly

### 9.1 Efficient Batch Assembly

When solving a set of linear systems that share the same EFIE matrix $\mathbf{Z}$ but have different RHS vectors (e.g., for multiple incident angles or port excitations), it is efficient to assemble all RHS vectors simultaneously. The function `assemble_multiple_excitations` does exactly that:

```julia
V = assemble_multiple_excitations(mesh, rwg, exc_list; quad_order=3)
```

where `exc_list` is a vector of `AbstractExcitation` objects. The returned `V` is a matrix of size `(N, M)` with `N = rwg.nedges` and `M = length(exc_list)`; column `j` contains the RHS vector for excitation `exc_list[j]`.

### 9.2 Implementation

The function loops over the excitations and calls `assemble_excitation` for each, storing the results as columns of `V`. This straightforward approach allows each excitation to be assembled with its own optimal quadrature order (if needed) and leverages any caching within individual assembly routines.

### 9.3 Use Cases

- **Monostatic RCS sweeps**: Compute scattering for many incident directions without reassembling $\mathbf{Z}$.
- **Multi‑port network analysis**: Exciting each port separately to fill a scattering‑matrix.
- **Optimization with multiple right‑hand sides**: Some design objectives depend on the response to several incident fields simultaneously.

### 9.4 Example

```julia
using DiffMoM

mesh = make_rect_plate(0.1, 0.1, 10, 10)
rwg = build_rwg(mesh)
k = 2π / 0.3

# List of plane waves at different angles
exc_list = [
    make_plane_wave(Vec3(0,0,-k), 1.0, Vec3(1,0,0)),
    make_plane_wave(Vec3(0,0,-k), 1.0, Vec3(0,1,0)),
    make_plane_wave(Vec3(-k,0,0), 1.0, Vec3(0,0,1))
]

V = assemble_multiple_excitations(mesh, rwg, exc_list; quad_order=3)
println("RHS matrix size: ", size(V))  # (N, 3)

# Verify each column matches individual assembly
for j in 1:length(exc_list)
    vj = assemble_excitation(mesh, rwg, exc_list[j]; quad_order=3)
    err = norm(V[:,j] - vj) / norm(vj)
    println("Column $j error: $err")
end
```

---

## 10. Validation Gates and Physical Interpretation

The package includes a comprehensive test suite for excitation functions. These tests are not mere software checks; they verify that the mathematical models are implemented correctly and produce physically meaningful results.

### 10.1 Plane‑Wave Consistency

- **Test**: Compare the RHS vector assembled via `PlaneWaveExcitation` with a direct quadrature of the analytical plane‑wave field.
- **Physical meaning**: Ensures that the discrete projection faithfully represents the continuous inner product.
- **Tolerance**: Machine precision ($\sim 10^{-15}$ relative error).

### 10.2 Delta‑Gap Scaling

- **Test**: Verify that the non‑zero entry of a delta‑gap excitation equals $V/g$ and that all other entries are zero.
- **Physical meaning**: Confirms the lumped‑gap approximation and checks that the implementation respects the support of the basis function.
- **Tolerance**: Exact within floating‑point rounding.

### 10.3 Dipole and Loop Far‑Field Patterns

- **Test**: Compare the far‑field pattern computed from a dipole/loop excitation with the analytical pattern (via `make_analytic_dipole_pattern_feed`).
- **Physical meaning**: Validates that the near‑field to far‑field transformation works correctly and that the source model radiates as expected.
- **Tolerance**: $< 10^{-6}$ relative error in amplitude, $< 10^{-4}$ rad in phase.

### 10.4 Cross‑Polarization and Phase Consistency

- **Test**: For an electric dipole oriented along $x$, the far‑field $E_\phi$ component should be negligible compared to $E_\theta$. For a loop (magnetic dipole) oriented along $z$, the opposite holds.
- **Physical meaning**: Checks that the vector polarization of the models aligns with theory. Also verifies the $90^\circ$ phase relationship between electric and magnetic dipoles at certain observation angles.
- **Tolerance**: Cross‑pol suppression $> 40$ dB, phase error $< 1^\circ$.

### 10.5 Imported‑Field Semantics

- **Test**: Compare `ImportedExcitation(kind=:electric_field)` with `ImportedExcitation(kind=:surface_current_density, eta_equiv=η0)` when the source function returns the same vector.
- **Physical meaning**: Ensures the two semantic variants produce identical RHS vectors when the mapping $\mathbf{E}^{\mathrm{inc}} = \eta_0 \mathbf{J}_s$ holds.
- **Tolerance**: Machine precision.

### 10.6 Pattern‑Feed Convention Handling

- **Test**: Create a pattern feed with `convention=:exp_minus_iwt` and verify that the field matches the conjugated coefficients of the `:exp_plus_iwt` case.
- **Physical meaning**: Guarantees correct phase handling when importing data from tools with the opposite time convention.
- **Tolerance**: Machine precision in the real and imaginary parts.

### 10.7 Multi‑Excitation Linearity

- **Test**: Compare `MultiExcitation` assembly with manual weighted sum of individual RHS vectors.
- **Physical meaning**: Confirms the linear‑superposition principle is preserved in the discrete implementation.
- **Tolerance**: Machine precision.

### 10.8 Multiple‑Excitations Matrix

- **Test**: Verify that each column of the matrix from `assemble_multiple_excitations` matches the corresponding individually assembled vector.
- **Physical meaning**: Ensures batch assembly is equivalent to sequential assembly.
- **Tolerance**: Machine precision.

These validation gates are run as part of the continuous‑integration pipeline. If you extend the excitation system (e.g., add a new excitation type), you should add corresponding tests to maintain the same level of physical verification.

---

## 11. Common Failure Modes and Remedies

### 11.1 Using `ImportedExcitation` as a Source Solver

**Symptom**: The imported field does not satisfy Maxwell’s equations, leading to unphysical scattering results or solver convergence issues.

**Cause**: Treating `ImportedExcitation` as a way to “inject” arbitrary currents without considering the radiation condition.

**Remedy**: Use `ImportedExcitation` only when you already have a **valid incident field** (e.g., from an analytic expression or a trusted external solver). If you need to radiate from a given current distribution, you should solve a separate source‑radiation problem (possibly with a volume IE or FEM solver) and import the resulting incident field.

### 11.2 Power‑Only Pattern Files

**Symptom**: `PatternFeedExcitation` produces incorrect phase or polarization.

**Cause**: Importing gain/directivity patterns that lack complex phase information.

**Remedy**: Ensure your pattern data include **complex $E_\theta$ and $E_\phi$ components**. If only power patterns are available, you can approximate phases (e.g., uniform phase) but this will affect interference phenomena.

### 11.3 Time‑Convention Mismatch

**Symptom**: Fields have the wrong phase progression, leading to errors in interference or frequency‑domain responses.

**Cause**: Importing data that assume $e^{-i\omega t}$ without proper conjugation.

**Remedy**: Use `convention=:exp_minus_iwt` in `make_pattern_feed`, or manually conjugate the coefficients before constructing the excitation.

### 11.4 Insufficient Quadrature Order

**Symptom**: The RHS vector changes noticeably when increasing `quad_order`, indicating that the quadrature is not capturing field variations accurately.

**Cause**: Rapidly varying incident field (e.g., high‑frequency oscillation or near‑field singularities) sampled with too few quadrature points.

**Remedy**: Increase `quad_order` in `assemble_excitation` or set a higher `min_quad_order` in `ImportedExcitation`. The supported orders are 1, 3, 4, 7; choose the smallest that gives converged results.

### 11.5 Wrong Phase Center for Pattern Feed

**Symptom**: The incident field has an unexpected spatial phase pattern.

**Cause**: The phase‑center position `phase_center` does not match the coordinate system used to define the pattern coefficients.

**Remedy**: Set `phase_center` to the location of the antenna’s phase reference (often the feed point or aperture center).

### 11.6 Comparing Only Scalar Power

**Symptom**: Two excitations appear equivalent in power but produce different scattering results.

**Cause**: Ignoring vector polarization and phase differences.

**Remedy**: Compare **complex vector fields** (or at least co‑pol and cross‑pol components) rather than just power/amplitude.

---

## 12. Code Map and Extension Guide

### 12.1 Primary Source File

All excitation‑related code resides in **`src/assembly/Excitation.jl`**. The main blocks are:

- Exports and `AbstractExcitation` base type.
- Excitation structs (`PlaneWaveExcitation`, `DipoleExcitation`, `LoopExcitation`, `ImportedExcitation`, `PatternFeedExcitation`, `DeltaGapExcitation`, `PortExcitation`, `MultiExcitation`).
- Constructor helpers (`make_plane_wave`, `make_dipole`, `make_loop`, `make_pattern_feed`, `make_imported_excitation`, `make_multi_excitation`).
- Field evaluators (`plane_wave_field`, `dipole_incident_field`, `loop_incident_field`, `pattern_feed_field`).
- Dispatch and assembly routines (`assemble_excitation`, `assemble_plane_wave`, `assemble_dipole`, `assemble_loop`, `assemble_imported_excitation`, `assemble_pattern_feed`, `assemble_port`, `assemble_delta_gap`, `assemble_multi`, `assemble_multiple_excitations`).

### 12.2 Adding a New Excitation Type

To extend the excitation system with a custom model, follow these steps:

1. **Define a struct** subtype of `AbstractExcitation` with the necessary fields.
2. **Implement a field‑evaluation function** that computes $\mathbf{E}^{\mathrm{inc}}(\mathbf{r})$ for your model.
3. **Add a specialized assembly function** (e.g., `assemble_my_excitation`) that performs the quadrature projection.
4. **Extend the dispatcher** `assemble_excitation` to handle your new type.
5. **Provide a constructor helper** (e.g., `make_my_excitation`) for user convenience.
6. **Add validation tests** to verify physical correctness (amplitude, polarization, phase, linearity).

Example skeleton:

```julia
struct MyExcitation <: AbstractExcitation
    param1::Float64
    param2::Vec3
    frequency::Float64
end

make_my_excitation(p1, p2, freq) = MyExcitation(p1, p2, freq)

function my_incident_field(r::Vec3, exc::MyExcitation)
    # Compute E_inc(r) here
    return CVec3(...)
end

function assemble_my_excitation(mesh::TriMesh, rwg::RWGData,
                                exc::MyExcitation; quad_order::Int=3)
    # Standard quadrature projection using my_incident_field
    N = rwg.nedges
    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)
    v = zeros(ComplexF64, N)
    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = triangle_area(mesh, t)
            pts = tri_quad_points(mesh, t, xi)
            for q in 1:Nq
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = my_incident_field(rq, exc)
                v[n] += -wq[q] * dot(fn, Einc) * (2 * A)
            end
        end
    end
    return v
end

# Extend the dispatcher
function assemble_excitation(mesh::TriMesh, rwg::RWGData,
                             exc::MyExcitation; quad_order::Int=3)
    return assemble_my_excitation(mesh, rwg, exc; quad_order=quad_order)
end
```

### 12.3 Related Example Scripts

Several example scripts demonstrate excitation usage:

- **`examples/07_pattern_feed.jl`**: Compares far‑field patterns of electric dipoles and loops, illustrating polarization differences.
- **`examples/07_pattern_feed.jl`**: Validates pattern‑feed generation against analytic dipole patterns.
- **`examples/07_pattern_feed.jl`**: Shows how to import measured/simulated horn patterns and use them as incident fields.
- **`examples/07_pattern_feed.jl`**: Utilities for converting radiation‑pattern data into the format required by `PatternFeedExcitation`.

These examples serve as both tutorials and templates for your own excitation workflows.

---

## 13. Summary and Best Practices

Excitation modeling in `DiffMoM.jl` is built on a unified principle: **construct a known incident field and project it onto the RWG basis to form the EFIE right‑hand side.** The variety of excitation types—plane wave, dipole, loop, delta‑gap, port, imported field, pattern feed—covers a wide range of practical scenarios while maintaining a consistent assembly pipeline.

### 13.1 Decision Tree for Choosing an Excitation

Use the following flowchart to select the appropriate excitation type:

1. **Do you know the incident field as an analytic expression?**
   - Yes → Use `ImportedExcitation(kind=:electric_field)`.
   - No → Proceed.

2. **Is the source a far‑field illuminator (e.g., radar, plane‑wave scattering)?**
   - Yes → Use `PlaneWaveExcitation`.
   - No → Proceed.

3. **Do you have complex far‑field pattern data ($E_\theta, E_\phi$) on a spherical grid?**
   - Yes → Use `PatternFeedExcitation`.
   - No → Proceed.

4. **Is the source a localized electric or magnetic dipole?**
   - Yes → Use `DipoleExcitation` or `LoopExcitation`.
   - No → Proceed.

5. **Is the feed modeled as a voltage across one or more RWG edges?**
   - Single edge → Use `DeltaGapExcitation`.
   - Multiple edges → Use `PortExcitation`.

6. **Are you combining multiple excitations?**
   - Yes → Use `MultiExcitation`.

If none of these fit, you may need to implement a custom excitation type following the extension guide in Section 12.2.

### 13.2 Verification Checklist

Before trusting results from a new excitation setup, run these sanity checks:

- [ ] **Quadrature convergence**: Increase `quad_order` by one level; the RHS vector should change by less than $10^{-6}$ relative.
- [ ] **Energy consistency**: For a PEC case, compare `input_power(I, v)` and `radiated_power(E_ff, grid)` (or `energy_ratio(I, v, E_ff, grid)`), which should be close to 1 in a converged run.
- [ ] **Far‑field pattern**: If applicable, compare the far‑field pattern with an independent calculation (analytical, another code).
- [ ] **Linear superposition**: Test that `MultiExcitation` matches manual combination.
- [ ] **Time‑convention**: Verify phase progression matches expectation (e.g., plane wave should advance phase in the direction of propagation).

### 13.3 Performance Considerations

- **Quadrature order**: Higher order increases assembly cost. Use the lowest order that gives converged results for your field variation.
- **Multiple excitations**: When solving for many RHS vectors, use `assemble_multiple_excitations` to avoid repeated mesh traversal.
- **Pattern‑feed interpolation**: Bilinear interpolation is fast but assumes the pattern is smooth on the grid scale. If the pattern has rapid angular variations, use a finer grid or higher‑order interpolation.
- **Imported‑field evaluation**: The user‑supplied function is called at every quadrature point; optimize it for speed if assembly time becomes significant.

### 13.4 Reproducibility

To ensure that your excitation setup can be reproduced later, record the following in your experiment logs:

- Excitation type and all parameters (e.g., `k_vec`, `E0`, `pol` for plane wave).
- Quadrature order used in assembly.
- For imported fields: the source function definition or a hash of the imported data file.
- For pattern feeds: grid parameters (`theta`, `phi`), convention, phase center.
- Any custom extensions or modifications to the excitation code.

By following these guidelines, you can leverage the flexibility of the excitation system while maintaining physical correctness and numerical robustness.

---

## 14. Further Reading

- **Chew, W. C., *Waves and Fields in Inhomogeneous Media*** (1995) – Chapter on plane‑wave expansions and dipole radiation.
- **Balanis, C. A., *Antenna Theory: Analysis and Design*** (2016) – Standard reference for dipole, loop, and pattern‑feed models.
- **Harrington, R. F., *Field Computation by Moment Methods*** (1993) – Classical text on MoM, including delta‑gap and port excitations.
- **IEEE Standard for Definitions of Terms for Antennas** (IEEE Std 145‑2013) – Clarifies polarization, pattern, and phase‑center definitions.

For implementation details, consult the Julia documentation on function types, multiple dispatch, and linear algebra, which underpin the excitation system’s design.

---
