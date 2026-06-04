# API: Density Topology

## Purpose

Reference for the density-based topology optimization API:

- Per-triangle material interpolation (SIMP-style impedance penalty).
- Density filtering and Heaviside projection.
- Adjoint gradients through the full chain `rho -> rho_tilde -> rho_bar -> J`.

This API is intended for design problems where each triangle carries a scalar density variable `rho_t in [0,1]`.

---

## `DensityConfig`

Configuration type for density interpolation and penalty scaling.

```julia
struct DensityConfig
    p::Float64          # SIMP penalization power
    Z_max::Float64      # void penalty impedance
    vf_target::Float64  # target metal volume fraction
end
```

### Constructor

```julia
DensityConfig(; p=3.0, Z_max_factor=1000.0, eta0=376.730313668, vf_target=0.5)
```

Constructs:

- `p`: SIMP power.
- `Z_max = Z_max_factor * eta0`.
- `vf_target`: target volume fraction.

---

## Density-Interpolated Penalty Assembly

### `precompute_triangle_mass(mesh, rwg; quad_order=3)`

Precompute per-triangle mass matrices:

```
M_t[m,n] = integral_t f_m . f_n dS
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Geometry mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `quad_order` | `Int` | `3` | Triangle quadrature order. |

**Returns:** `Vector{SparseMatrixCSC{Float64,Int}}` of length `Nt`, one sparse matrix per triangle.

Only basis functions supported on triangle `t` contribute to `M_t`.

---

### `assemble_Z_penalty(Mt, rho_bar, config)`

Assemble the density penalty matrix:

```
Z_penalty = sum_t (1 - rho_bar[t]^p) * Z_max * M_t
```

with `p = config.p`, `Z_max = config.Z_max`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Mt` | `Vector{<:AbstractMatrix}` | Triangle mass matrices from `precompute_triangle_mass`. |
| `rho_bar` | `AbstractVector{<:Real}` | Projected densities, length `Nt`. |
| `config` | `DensityConfig` | Density optimization configuration. |

**Returns:** `Matrix{ComplexF64}` dense penalty matrix.

Behavior:
- `rho_bar[t] = 1`: no penalty on triangle `t`.
- `rho_bar[t] = 0`: maximum penalty `Z_max * M_t`.

---

### `assemble_dZ_drhobar(Mt, rho_bar, config, t)`

Derivative of the penalty matrix with respect to one projected density:

```
dZ/drho_bar[t] = -p * rho_bar[t]^(p-1) * Z_max * M_t
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Mt` | `Vector{<:AbstractMatrix}` | Triangle mass matrices. |
| `rho_bar` | `AbstractVector{<:Real}` | Projected densities. |
| `config` | `DensityConfig` | Density configuration. |
| `t` | `Int` | Triangle index (1-based). |

**Returns:** Matrix with the same shape as `M_t`.

---

## Filtering and Projection

### `build_filter_weights(mesh, r_min)`

Build the sparse conic filter weights:

```
W[t,s] = max(0, r_min - ||c_t - c_s||)
```

where `c_t` are triangle centroids.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mesh` | `TriMesh` | Mesh with `Nt` triangles. |
| `r_min` | `Float64` | Filter radius (meters). |

**Returns:** Tuple `(W, w_sum)` where:
- `W::SparseMatrixCSC{Float64,Int}` (size `Nt x Nt`)
- `w_sum::Vector{Float64}` with `w_sum[t] = sum_s W[t,s]`

---

### `apply_filter(W, w_sum, rho)`

Apply density filtering:

```
rho_tilde = (W * rho) ./ w_sum
```

**Returns:** Filtered density vector `rho_tilde`.

---

### `apply_filter_transpose(W, w_sum, g_rho_tilde)`

Adjoint (transpose) of the filtering map, used in gradient backpropagation:

```
g_rho = W' * (g_rho_tilde ./ w_sum)
```

**Returns:** Gradient with respect to raw densities `rho`.

---

### `heaviside_project(rho_tilde, beta, eta=0.5)`

Smooth Heaviside projection:

```
rho_bar = [tanh(beta*eta) + tanh(beta*(rho_tilde - eta))] /
          [tanh(beta*eta) + tanh(beta*(1-eta))]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rho_tilde` | `AbstractVector` | -- | Filtered densities. |
| `beta` | `Real` | -- | Sharpness parameter. |
| `eta` | `Real` | `0.5` | Threshold parameter. |

**Returns:** Projected densities `rho_bar`.

---

### `heaviside_derivative(rho_tilde, beta, eta=0.5)`

Derivative of the smooth Heaviside map:

```
dH/drho_tilde =
beta * (1 - tanh(beta*(rho_tilde-eta))^2) /
[tanh(beta*eta) + tanh(beta*(1-eta))]
```

**Returns:** Vector of elementwise derivatives.

---

### `filter_and_project(W, w_sum, rho, beta, eta=0.5)`

Run the full pipeline:

```
rho_tilde = apply_filter(W, w_sum, rho)
rho_bar   = heaviside_project(rho_tilde, beta, eta)
```

**Returns:** Tuple `(rho_tilde, rho_bar)`.

---

### `gradient_chain_rule(g_rho_bar, rho_tilde, W, w_sum, beta, eta=0.5)`

Backpropagate a gradient through projection and filtering:

```
g_rho_tilde = g_rho_bar .* heaviside_derivative(rho_tilde, beta, eta)
g_rho       = apply_filter_transpose(W, w_sum, g_rho_tilde)
```

**Returns:** Gradient with respect to raw design variables `rho`.

---

## Adjoint Gradients for Density Designs

### `gradient_density(Mt, I, lambda, rho_bar, config)`

Compute gradient with respect to projected densities:

```
g[t] = -2 * Re( lambda' * (dZ/drho_bar[t]) * I )
```

with `dZ/drho_bar[t]` from `assemble_dZ_drhobar`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Mt` | `Vector{<:AbstractMatrix}` | Triangle mass matrices. |
| `I` | `Vector{<:Number}` | Forward current coefficients. |
| `lambda` | `Vector{<:Number}` | Adjoint variable. |
| `rho_bar` | `AbstractVector{<:Real}` | Projected densities. |
| `config` | `DensityConfig` | Density configuration. |

**Returns:** `Vector{Float64}` with one entry per triangle.

---

### `gradient_density_full(Mt, I, lambda, rho_tilde, rho_bar, config, W, w_sum, beta; eta=0.5)`

Full gradient with chain rule to raw densities:

1. `g_rho_bar = gradient_density(...)`
2. `g_rho = gradient_chain_rule(g_rho_bar, rho_tilde, W, w_sum, beta, eta)`

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Mt` | `Vector{<:AbstractMatrix}` | Triangle mass matrices. |
| `I` | `Vector{<:Number}` | Forward coefficients. |
| `lambda` | `Vector{<:Number}` | Adjoint coefficients. |
| `rho_tilde` | `AbstractVector{<:Real}` | Filtered densities. |
| `rho_bar` | `AbstractVector{<:Real}` | Projected densities. |
| `config` | `DensityConfig` | Density configuration. |
| `W` | `AbstractSparseMatrix` | Filter matrix. |
| `w_sum` | `AbstractVector` | Filter row sums. |
| `beta` | `Real` | Projection sharpness. |
| `eta` | `Real` | Projection threshold (`0.5` default). |

**Returns:** `Vector{Float64}` gradient with respect to raw `rho`.

---

## Typical Pipeline

```julia
cfg = DensityConfig(; p=3.0, Z_max_factor=1000.0, vf_target=0.5)

Mt = precompute_triangle_mass(mesh, rwg)
W, w_sum = build_filter_weights(mesh, r_min)

rho_tilde, rho_bar = filter_and_project(W, w_sum, rho, beta, eta)
Z_pen = assemble_Z_penalty(Mt, rho_bar, cfg)
Z = Z_efie + Z_pen

I = solve_forward(Z, v)
lambda = solve_adjoint(Z, Q, I)

g_rho = gradient_density_full(Mt, I, lambda, rho_tilde, rho_bar, cfg, W, w_sum, beta; eta=eta)
```

---

## Notes

- `build_filter_weights` is O(`Nt^2`) in both work and temporary pair checks, so very fine meshes can be expensive without spatial acceleration.
- `precompute_triangle_mass` returns one sparse matrix per triangle; memory grows with both RWG count and local support overlap.
- The penalty model uses real-valued `Z_max`; large values improve material contrast but can worsen conditioning.
- `gradient_density_full` assumes `rho_tilde` and `rho_bar` come from the same `W`, `w_sum`, `beta`, `eta` pipeline used in the forward pass.

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/assembly/DensityInterpolation.jl` | `DensityConfig`, `precompute_triangle_mass`, `assemble_Z_penalty`, `assemble_dZ_drhobar` |
| `src/optimization/DensityFiltering.jl` | `build_filter_weights`, `apply_filter`, `apply_filter_transpose`, `heaviside_project`, `heaviside_derivative`, `filter_and_project`, `gradient_chain_rule` |
| `src/optimization/DensityAdjoint.jl` | `gradient_density`, `gradient_density_full` |

---

## Exercises

- **Basic:** Build `W, w_sum` for a small mesh and verify `filter_and_project` returns vectors of length `ntriangles(mesh)`.
- **Practical:** For a fixed `rho`, sweep `beta = 1, 4, 16, 64` and confirm `rho_bar` becomes increasingly binary.
- **Challenge:** Verify `gradient_density_full` against central finite differences on 5--10 random density variables and report max relative error.
