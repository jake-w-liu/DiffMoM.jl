# API: 3D Material Models

## Purpose

Reference for the pure constitutive material models consumed by the 3D volume
solver (see [dda-volume-3d.md](dda-volume-3d.md)). These types and helper
functions describe relative permittivity, relative permeability, combined
magnetodielectric media, and bianisotropic media for volumetric scatterers, and
they provide frequency evaluators for static and dispersive (Drude / Lorentz /
Debye) responses.

All models follow the package's `exp(+i omega t)` convention. With this
convention, passive electric or magnetic loss corresponds to a non-positive
imaginary material response (`imag(eps_r) <= 0`, `imag(mu_r) <= 0`). The model
constructors and dispersive evaluators validate passivity by default; pass
`passive=false` to bypass these checks (for example, to model gain media).

The models split into four groups:

- **Permittivity models:** `IsotropicMaterial3D`, `DiagonalAnisotropicMaterial3D`,
  `TensorAnisotropicMaterial3D`, and the dispersive `DrudePermittivity3D`,
  `LorentzPermittivity3D`, `DebyePermittivity3D`.
- **Permeability models:** `IsotropicPermeability3D`, `DiagonalPermeability3D`,
  `TensorPermeability3D`.
- **Magnetodielectric and bianisotropic media:** `MagneticMaterial3D`,
  `BianisotropicMaterial3D`.
- **Evaluator functions:** `material_epsr_3d`, `material_mur_3d`,
  `material_bianisotropic_matrix_3d`, and the standalone dispersive evaluators
  `drude_epsr_3d`, `lorentz_epsr_3d`, `debye_epsr_3d`.

**See also:** for the theory and a worked walkthrough, see
[Material Models: Dispersive and Anisotropic Media](../formulations/03-material-models.md).

---

## Permittivity Models

### `IsotropicMaterial3D(eps_r; passive=true)`

Static isotropic relative permittivity model: a single scalar relative
permittivity used for every direction.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps_r` | Number (coerced to `ComplexF64`) | -- | Relative permittivity. Must be finite. |
| `passive` | `Bool` | `true` | If `true`, require `imag(eps_r) <= 0` (passive loss under `exp(+i omega t)`). |

**Field:** `eps_r::ComplexF64`.

**Returns:** an `IsotropicMaterial3D` instance.

```julia
iso = IsotropicMaterial3D(2.5 - 0.1im)
material_epsr_3d(iso, 1.0e9)   # 2.5 - 0.1im
```

---

### `DiagonalAnisotropicMaterial3D((eps_x, eps_y, eps_z); passive=true)`

Static diagonal anisotropic relative permittivity model. The principal-axis
permittivities are stored as a 3-vector; off-diagonal coupling is zero.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps_r` | 3-element collection (coerced to `SVector{3,ComplexF64}`) | -- | Diagonal permittivities `(eps_x, eps_y, eps_z)`. Must have exactly three finite entries. |
| `passive` | `Bool` | `true` | If `true`, require `imag(eps_r[a]) <= 0` for each component. |

**Field:** `eps_r::SVector{3,ComplexF64}`.

**Returns:** a `DiagonalAnisotropicMaterial3D` instance.

```julia
diag = DiagonalAnisotropicMaterial3D((2.0 - 0.1im, 3.0 - 0.2im, 4.0 + 0.0im))
material_epsr_3d(diag, 3.0)   # SVector{3,ComplexF64}
```

---

### `TensorAnisotropicMaterial3D(eps_r; passive=true)`

Static full 3x3 relative permittivity tensor model, allowing off-diagonal
coupling between Cartesian field components.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps_r` | 3x3 array (coerced to `SMatrix{3,3,ComplexF64,9}`) | -- | Permittivity tensor. Must be `(3, 3)` with finite entries. |
| `passive` | `Bool` | `true` | If `true`, require the anti-Hermitian loss matrix `(eps - eps') / (2im)` to be negative semidefinite. |

**Field:** `eps_r::SMatrix{3,3,ComplexF64,9}`.

**Returns:** a `TensorAnisotropicMaterial3D` instance.

```julia
tensor = TensorAnisotropicMaterial3D(ComplexF64[
    2.0 - 0.1im 0.0 + 0.0im 0.0 + 0.0im
    0.0 + 0.0im 3.0 - 0.2im 0.0 + 0.0im
    0.0 + 0.0im 0.0 + 0.0im 4.0 - 0.3im
])
size(material_epsr_3d(tensor, 3.0))   # (3, 3)
```

---

### `DrudePermittivity3D(eps_inf, plasma_freq_hz, gamma_hz; passive=true)`

Dispersive Drude relative permittivity model (free-electron / plasma response):

```
eps = eps_inf - omega_p^2 / (omega^2 - i gamma omega)
```

where `omega = 2*pi*freq_hz`, `omega_p = 2*pi*plasma_freq_hz`, and
`gamma = 2*pi*gamma_hz`. The permittivity is frequency dependent and is
evaluated via `material_epsr_3d(model, freq_hz)` or `drude_epsr_3d`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps_inf` | Number (coerced to `ComplexF64`) | -- | High-frequency permittivity limit. Must be finite. |
| `plasma_freq_hz` | Real (coerced to `Float64`) | -- | Plasma frequency in Hz. Must be finite and nonnegative. |
| `gamma_hz` | Real (coerced to `Float64`) | -- | Collision / damping rate in Hz. Must be finite and nonnegative. |
| `passive` | `Bool` | `true` | If `true`, require `imag(eps_inf) <= 0`. |

**Fields:** `eps_inf::ComplexF64`, `plasma_freq_hz::Float64`, `gamma_hz::Float64`.

**Returns:** a `DrudePermittivity3D` instance.

```julia
drude = DrudePermittivity3D(1.0, 1.0e15, 1.0e13)
material_epsr_3d(drude, 2.0e14)   # ComplexF64, frequency-dependent
```

---

### `LorentzPermittivity3D(eps_inf, strength, resonance_freq_hz, gamma_hz; passive=true)`

Dispersive Lorentz relative permittivity model (single bound-charge resonance):

```
eps = eps_inf + strength * omega_0^2 / (omega_0^2 - omega^2 + i gamma omega)
```

where `omega = 2*pi*freq_hz`, `omega_0 = 2*pi*resonance_freq_hz`, and
`gamma = 2*pi*gamma_hz`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps_inf` | Number (coerced to `ComplexF64`) | -- | High-frequency permittivity limit. Must be finite. |
| `strength` | Number (coerced to `ComplexF64`) | -- | Oscillator strength. Must be finite. |
| `resonance_freq_hz` | Real (coerced to `Float64`) | -- | Resonance frequency in Hz. Must be finite and positive. |
| `gamma_hz` | Real (coerced to `Float64`) | -- | Damping rate in Hz. Must be finite and nonnegative. |
| `passive` | `Bool` | `true` | If `true`, require `imag(eps_inf) <= 0`, `imag(strength) <= 0`, and `real(strength) >= 0`. |

**Fields:** `eps_inf::ComplexF64`, `strength::ComplexF64`,
`resonance_freq_hz::Float64`, `gamma_hz::Float64`.

**Returns:** a `LorentzPermittivity3D` instance.

```julia
lorentz = LorentzPermittivity3D(1.0, 0.5, 2.0e14, 1.0e13)
material_epsr_3d(lorentz, 1.0e14)   # ComplexF64, frequency-dependent
```

---

### `DebyePermittivity3D(eps_static, eps_inf, tau_s; passive=true)`

Dispersive Debye relaxation model (orientational polarization):

```
eps = eps_inf + (eps_static - eps_inf) / (1 + i omega tau)
```

where `omega = 2*pi*freq_hz` and `tau = tau_s`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps_static` | Number (coerced to `ComplexF64`) | -- | Static (zero-frequency) permittivity. Must be finite. |
| `eps_inf` | Number (coerced to `ComplexF64`) | -- | High-frequency permittivity limit. Must be finite. |
| `tau_s` | Real (coerced to `Float64`) | -- | Relaxation time in seconds. Must be finite and positive. |
| `passive` | `Bool` | `true` | If `true`, require `imag(eps_static) <= 0`, `imag(eps_inf) <= 0`, and `real(eps_static - eps_inf) >= 0`. |

**Fields:** `eps_static::ComplexF64`, `eps_inf::ComplexF64`, `tau_s::Float64`.

**Returns:** a `DebyePermittivity3D` instance.

```julia
debye = DebyePermittivity3D(4.0, 2.0, 1.0e-10)
material_epsr_3d(debye, 1.0e9)   # ComplexF64, frequency-dependent
```

---

## Permeability Models

### `IsotropicPermeability3D(mu_r; passive=true)`

Static isotropic relative permeability model. Passive magnetic loss follows the
same `exp(+i omega t)` sign convention as permittivity.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu_r` | Number (coerced to `ComplexF64`) | -- | Relative permeability. Must be finite. |
| `passive` | `Bool` | `true` | If `true`, require `imag(mu_r) <= 0`. |

**Field:** `mu_r::ComplexF64`.

**Returns:** an `IsotropicPermeability3D` instance.

```julia
mu = IsotropicPermeability3D(1.2 - 0.05im)
material_mur_3d(mu, 1.0e9)   # 1.2 - 0.05im
```

---

### `DiagonalPermeability3D((mu_x, mu_y, mu_z); passive=true)`

Static diagonal anisotropic relative permeability model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu_r` | 3-element collection (coerced to `SVector{3,ComplexF64}`) | -- | Diagonal permeabilities `(mu_x, mu_y, mu_z)`. Must have exactly three finite entries. |
| `passive` | `Bool` | `true` | If `true`, require `imag(mu_r[a]) <= 0` for each component. |

**Field:** `mu_r::SVector{3,ComplexF64}`.

**Returns:** a `DiagonalPermeability3D` instance.

```julia
mu_diag = DiagonalPermeability3D((1.2 - 0.01im, 1.4 - 0.02im, 1.0 + 0.0im))
material_mur_3d(mu_diag, 2.0)   # SVector{3,ComplexF64}
```

---

### `TensorPermeability3D(mu_r; passive=true)`

Static full 3x3 relative permeability tensor model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu_r` | 3x3 array (coerced to `SMatrix{3,3,ComplexF64,9}`) | -- | Permeability tensor. Must be `(3, 3)` with finite entries. |
| `passive` | `Bool` | `true` | If `true`, require the anti-Hermitian loss matrix `(mu - mu') / (2im)` to be negative semidefinite. |

**Field:** `mu_r::SMatrix{3,3,ComplexF64,9}`.

**Returns:** a `TensorPermeability3D` instance.

```julia
mu_tensor = TensorPermeability3D(ComplexF64[
    1.2 - 0.01im 0.0 + 0.0im 0.0 + 0.0im
    0.0 + 0.0im 1.4 - 0.02im 0.0 + 0.0im
    0.0 + 0.0im 0.0 + 0.0im 1.0 + 0.0im
])
size(material_mur_3d(mu_tensor, 2.0))   # (3, 3)
```

---

## Magnetodielectric and Bianisotropic Media

### `MagneticMaterial3D(eps_model, mu_model)`

Combined magnetodielectric medium that pairs a permittivity model with a
permeability model. `material_epsr_3d` on a `MagneticMaterial3D` delegates to its
`eps_model`, and `material_mur_3d` delegates to its `mu_model`, so either field
may itself be any supported permittivity or permeability model (including
dispersive ones).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps_model` | any permittivity model or `Number` | -- | Permittivity model, stored in field `eps_model`. Evaluated by `material_epsr_3d`. |
| `mu_model` | any permeability model or `Number` | -- | Permeability model, stored in field `mu_model`. Evaluated by `material_mur_3d`. |

**Fields:** `eps_model::TE`, `mu_model::TM` (the type parameters are inferred
from the two supplied models).

**Returns:** a `MagneticMaterial3D` instance.

```julia
iso = IsotropicMaterial3D(2.5 - 0.1im)
mu  = IsotropicPermeability3D(1.2 - 0.05im)
magnetic = MagneticMaterial3D(iso, mu)
material_epsr_3d(magnetic, 1.0e9)   # delegates to iso
material_mur_3d(magnetic, 1.0e9)    # 1.2 - 0.05im
```

---

### `BianisotropicMaterial3D(C6; passive=true)`

Static normalized bianisotropic constitutive matrix. `C6` is a 6x6 tensor that
acts on the normalized field pair `[E; eta0*H]`, so its electric and magnetic
diagonal 3x3 blocks are the relative permittivity and permeability for uncoupled
media, while the off-diagonal blocks describe magnetoelectric coupling. The
volume DDA path (see [dda-volume-3d.md](dda-volume-3d.md)) converts this
normalized constitutive tensor to the solver's `[E; H] -> [q; m]` polarizability
convention.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `C6` | 6x6 array (coerced to `SMatrix{6,6,ComplexF64,36}`) | -- | Normalized bianisotropic constitutive matrix. Must be `(6, 6)` with finite entries. |
| `passive` | `Bool` | `true` | If `true`, require the anti-Hermitian loss matrix `(C6 - C6') / (2im)` to be negative semidefinite. |

**Field:** `C6::SMatrix{6,6,ComplexF64,36}`.

**Returns:** a `BianisotropicMaterial3D` instance.

```julia
C6 = Matrix{ComplexF64}(I, 6, 6)
C6[1, 1] = 2.0 - 0.01im
C6[4, 4] = 1.3 - 0.02im
C6[1, 5] = 0.05 + 0.0im
C6[5, 1] = 0.05 + 0.0im
bianiso = BianisotropicMaterial3D(C6)
material_bianisotropic_matrix_3d(bianiso, 2.0)   # bianiso.C6
```

---

## Evaluator Functions

### `material_epsr_3d(model, freq_hz_or_k0)`

Evaluate a material's relative permittivity at a frequency scale. Static models
(`Number`, `IsotropicMaterial3D`, `DiagonalAnisotropicMaterial3D`,
`TensorAnisotropicMaterial3D`) ignore the frequency scale except for finite
nonnegative validation; dispersive models (`DrudePermittivity3D`,
`LorentzPermittivity3D`, `DebyePermittivity3D`) interpret the argument as
frequency in Hz. A `MagneticMaterial3D` delegates to its `eps_model`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | permittivity model or `Number` | -- | Material model to evaluate. |
| `freq_hz_or_k0` | Real (coerced to `Float64`) | -- | Frequency in Hz (dispersive models) or a frequency/wavenumber scale (static models). Must be finite and nonnegative. |

**Returns:** `ComplexF64` for scalar models, `SVector{3,ComplexF64}` for diagonal
models, or `SMatrix{3,3,ComplexF64,9}` for tensor models.

```julia
iso = IsotropicMaterial3D(2.5 - 0.1im)
material_epsr_3d(iso, 1.0e9)   # 2.5 - 0.1im
```

---

### `material_mur_3d(model, freq_hz_or_k0)`

Evaluate a material's relative permeability at a frequency scale. Static models
(`Number`, `IsotropicPermeability3D`, `DiagonalPermeability3D`,
`TensorPermeability3D`) ignore the frequency scale except for finite nonnegative
validation. A `MagneticMaterial3D` delegates to its `mu_model`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | permeability model or `Number` | -- | Material model to evaluate. |
| `freq_hz_or_k0` | Real (coerced to `Float64`) | -- | Frequency / wavenumber scale. Must be finite and nonnegative. |

**Returns:** `ComplexF64` for scalar models, `SVector{3,ComplexF64}` for diagonal
models, or `SMatrix{3,3,ComplexF64,9}` for tensor models.

```julia
mu = IsotropicPermeability3D(1.2 - 0.05im)
material_mur_3d(mu, 1.0e9)   # 1.2 - 0.05im
```

---

### `material_bianisotropic_matrix_3d(model, freq_hz_or_k0)`

Evaluate the normalized static bianisotropic 6x6 material matrix. The frequency
scale is validated (finite and nonnegative) but otherwise ignored for the static
`BianisotropicMaterial3D` model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BianisotropicMaterial3D` | -- | Bianisotropic material model. |
| `freq_hz_or_k0` | Real (coerced to `Float64`) | -- | Frequency / wavenumber scale. Must be finite and nonnegative. |

**Returns:** `SMatrix{6,6,ComplexF64,36}` (the model's `C6`).

```julia
bianiso = BianisotropicMaterial3D(Matrix{ComplexF64}(I, 6, 6))
material_bianisotropic_matrix_3d(bianiso, 2.0)   # 6x6 SMatrix
```

---

### `drude_epsr_3d(freq_hz; eps_inf=1.0, plasma_freq_hz, gamma_hz, passive=true)`

Evaluate a Drude relative permittivity directly from parameters for the
`exp(+i omega t)` convention:

```
eps = eps_inf - omega_p^2 / (omega^2 - i gamma omega)
```

with `omega = 2*pi*freq_hz`, `omega_p = 2*pi*plasma_freq_hz`, and
`gamma = 2*pi*gamma_hz`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freq_hz` | Real (coerced to `Float64`) | -- | Evaluation frequency in Hz. Must be finite and positive. |
| `eps_inf` | Number (coerced to `ComplexF64`) | `1.0` | High-frequency permittivity limit. Must be finite. |
| `plasma_freq_hz` | Real (coerced to `Float64`) | -- (required keyword) | Plasma frequency in Hz. Must be finite and nonnegative. |
| `gamma_hz` | Real (coerced to `Float64`) | -- (required keyword) | Collision / damping rate in Hz. Must be finite and nonnegative. |
| `passive` | `Bool` | `true` | If `true`, require the resulting `imag(eps_r) <= 0`. |

**Returns:** `ComplexF64` relative permittivity.

```julia
eps = drude_epsr_3d(2.0e14; eps_inf=1.0, plasma_freq_hz=1.0e15, gamma_hz=1.0e13)
imag(eps) <= 0   # true (passive)
```

---

### `lorentz_epsr_3d(freq_hz; eps_inf=1.0, strength, resonance_freq_hz, gamma_hz, passive=true)`

Evaluate a Lorentz relative permittivity directly from parameters for the
`exp(+i omega t)` convention:

```
eps = eps_inf + strength * omega_0^2 / (omega_0^2 - omega^2 + i gamma omega)
```

with `omega = 2*pi*freq_hz`, `omega_0 = 2*pi*resonance_freq_hz`, and
`gamma = 2*pi*gamma_hz`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freq_hz` | Real (coerced to `Float64`) | -- | Evaluation frequency in Hz. Must be finite and nonnegative. |
| `eps_inf` | Number (coerced to `ComplexF64`) | `1.0` | High-frequency permittivity limit. Must be finite. |
| `strength` | Number (coerced to `ComplexF64`) | -- (required keyword) | Oscillator strength. Must be finite. |
| `resonance_freq_hz` | Real (coerced to `Float64`) | -- (required keyword) | Resonance frequency in Hz. Must be finite and positive. |
| `gamma_hz` | Real (coerced to `Float64`) | -- (required keyword) | Damping rate in Hz. Must be finite and nonnegative. |
| `passive` | `Bool` | `true` | If `true`, require the resulting `imag(eps_r) <= 0`. |

**Returns:** `ComplexF64` relative permittivity.

```julia
eps = lorentz_epsr_3d(1.0e14; eps_inf=1.0, strength=0.5,
                      resonance_freq_hz=2.0e14, gamma_hz=1.0e13)
imag(eps) <= 0   # true (passive)
```

---

### `debye_epsr_3d(freq_hz; eps_static, eps_inf=1.0, tau_s, passive=true)`

Evaluate a Debye relative permittivity directly from parameters for the
`exp(+i omega t)` convention:

```
eps = eps_inf + (eps_static - eps_inf) / (1 + i omega tau)
```

with `omega = 2*pi*freq_hz` and `tau = tau_s`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freq_hz` | Real (coerced to `Float64`) | -- | Evaluation frequency in Hz. Must be finite and nonnegative. |
| `eps_static` | Number (coerced to `ComplexF64`) | -- (required keyword) | Static (zero-frequency) permittivity. Must be finite. |
| `eps_inf` | Number (coerced to `ComplexF64`) | `1.0` | High-frequency permittivity limit. Must be finite. |
| `tau_s` | Real (coerced to `Float64`) | -- (required keyword) | Relaxation time in seconds. Must be finite and positive. |
| `passive` | `Bool` | `true` | If `true`, require the resulting `imag(eps_r) <= 0`. |

**Returns:** `ComplexF64` relative permittivity.

```julia
eps = debye_epsr_3d(1.0e9; eps_static=4.0, eps_inf=2.0, tau_s=1.0e-10)
imag(eps) <= 0   # true (passive)
```

---

## Code Mapping

| Symbol | Source File | Primary Users |
|--------|-------------|---------------|
| All permittivity, permeability, magnetodielectric, and bianisotropic models | `src/mom3d/MaterialModels3D.jl` | 3D volume DDA solver (see [dda-volume-3d.md](dda-volume-3d.md)) |
| `material_epsr_3d`, `material_mur_3d`, `material_bianisotropic_matrix_3d` | `src/mom3d/MaterialModels3D.jl` | Constitutive evaluation in the volume solver |
| `drude_epsr_3d`, `lorentz_epsr_3d`, `debye_epsr_3d` | `src/mom3d/MaterialModels3D.jl` | Standalone dispersive permittivity evaluation |

---

## Exercises

- **Basic:** Construct an `IsotropicMaterial3D(2.5 - 0.1im)` and confirm
  `material_epsr_3d` returns it unchanged at any finite nonnegative frequency.
- **Practical:** Build a `MagneticMaterial3D` from an `IsotropicMaterial3D` and an
  `IsotropicPermeability3D`, then verify `material_epsr_3d` and `material_mur_3d`
  delegate to the correct sub-model.
- **Challenge:** Sweep `drude_epsr_3d` across a frequency band and confirm
  `imag(eps_r) <= 0` everywhere (passivity). Then call it with `passive=false`
  and an active `eps_inf` to model gain.
