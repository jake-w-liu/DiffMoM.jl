# Multi-Angle RCS Optimization

## Purpose

This chapter extends the single-objective optimization framework from Chapters 1--4 to **multi-angle monostatic RCS minimization**: simultaneously reducing backscatter radar cross section over multiple incidence angles. This is the core requirement for practical stealth and scattering control design, where a single-angle optimum often shifts energy to unprotected angles rather than truly reducing scattering.

The chapter also covers the integration of **fast operators** (MLFMA, ACA) into the optimization loop via the `ImpedanceLoadedOperator` composite operator, enabling optimization of large-scale problems (N > 10,000) that are intractable with dense methods.

---

## Learning Goals

After this chapter, you should be able to:

1. Formulate a multi-angle RCS objective and understand why single-angle optimization is insufficient for practical RCS reduction.
2. Use `build_multiangle_configs` to set up multi-angle optimization problems.
3. Run `optimize_multiangle_rcs` with MLFMA or ACA base operators for large problems.
4. Use `direction_mask` to define backscatter cones around arbitrary incidence directions.
5. Interpret multi-angle optimization traces and validate results.

---

## 1. Why Multi-Angle Optimization?

### 1.1 The Single-Angle Problem

A single-angle optimization minimizes:

```
J_single(theta) = I_a^dagger Q_a I_a
```

where $Q_a$ selects far-field power near the backscatter direction for incidence angle $a$. The optimizer is free to redistribute scattered energy to other directions — the RCS at the optimized angle drops, but RCS at other angles may increase dramatically.

### 1.2 The Multi-Angle Objective

The multi-angle objective sums over $M$ incidence angles:

```
J(theta) = sum_{a=1}^{M} w_a (I_a^dagger Q_a I_a)
```

where:
- $I_a = Z(\theta)^{-1} v_a$ is the surface current for incidence angle $a$
- $Q_a$ selects the backscatter direction for angle $a$
- $w_a$ is the weight for angle $a$ (uniform by default)

The gradient is computed via the adjoint method, requiring $M$ forward solves and $M$ adjoint solves per iteration — but **no finite differences** regardless of the number of impedance parameters $P$:

```
g[p] = sum_{a=1}^{M} w_a * 2 Re{ lambda_a^dagger M_p I_a }
```

where $Z(\theta)^\dagger \lambda_a = Q_a I_a$.

### 1.3 Cost Per Iteration

| Component | Operations | Scaling |
|-----------|-----------|---------|
| Forward solves | M GMRES solves with composite operator | M * (cost of Z * x) * (GMRES iters) |
| Adjoint solves | M GMRES solves with adjoint operator | Same as forward |
| Gradient | P inner products per angle | M * P * O(N) |
| Line search | M forward solves per trial step | Typically 1--3 trial steps |

For MLFMA with O(N log N) matvec cost, the total per-iteration cost is O(M * N log N * GMRES_iters).

---

## 2. Problem Setup

### 2.1 The `direction_mask` Function

Unlike `cap_mask` (which only selects directions near the +z axis), `direction_mask` creates a cone mask around **any** direction. This is essential for multi-angle optimization where each backscatter direction differs.

```julia
mask = direction_mask(grid, direction; half_angle=pi/18)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `SphGrid` | -- | Spherical grid. |
| `direction` | `Vec3` | -- | Center direction of the cone (will be normalized). |
| `half_angle` | `Float64` | `pi/18` (~10 deg) | Half-angle of the cone in radians. |

**Returns:** `BitVector` of length `N_omega` with `true` where `dot(rhat_q, direction) >= cos(half_angle)`.

**Example:**

```julia
grid = make_sph_grid(90, 36)

# Backscatter mask for incidence from +z (same as cap_mask with theta_max=pi-10deg)
mask_z = direction_mask(grid, Vec3(0, 0, -1); half_angle=10*pi/180)

# Backscatter mask for 45-degree incidence in xz-plane
k_hat = Vec3(sin(pi/4), 0, cos(pi/4))
mask_45 = direction_mask(grid, -k_hat; half_angle=10*pi/180)
```

### 2.2 The `AngleConfig` Type

Each incidence angle in the multi-angle problem is represented by an `AngleConfig`:

```julia
struct AngleConfig
    k_vec::Vec3                     # Incidence wave vector (rad/m)
    pol::Vec3                       # Polarization (unit vector)
    v::Vector{ComplexF64}           # Pre-assembled excitation vector
    Q::Matrix{ComplexF64}           # Backscatter Q matrix for this angle
    weight::Float64                 # Weight in composite objective
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `k_vec` | `Vec3` | Wave vector $\mathbf{k} = k \hat{\mathbf{k}}$ where $\hat{\mathbf{k}}$ is the incidence direction. |
| `pol` | `Vec3` | Polarization unit vector (perpendicular to `k_vec`). |
| `v` | `Vector{ComplexF64}` | Excitation vector (pre-assembled by `assemble_excitation`). |
| `Q` | `Matrix{ComplexF64}` | Backscatter Q-matrix targeting the direction $-\hat{\mathbf{k}}$ with the specified cone angle. |
| `weight` | `Float64` | Weight $w_a$ in the composite objective. |

### 2.3 Building Configurations with `build_multiangle_configs`

```julia
configs = build_multiangle_configs(mesh, rwg, k, angles;
                                    grid=grid,
                                    backscatter_cone=10.0)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `k` | `Float64` | -- | Wavenumber (rad/m). |
| `angles` | `Vector{<:NamedTuple}` | -- | Incidence angle specifications (see below). |
| `grid` | `SphGrid` | -- | Spherical grid (keyword argument, shared across all angles). |
| `backscatter_cone` | `Float64` | `10.0` | Half-angle of backscatter mask in degrees. |

**Angle specification format:**

Each element of `angles` is a named tuple with fields:

| Field | Type | Description |
|-------|------|-------------|
| `theta_inc` | `Float64` | Polar angle of incidence (radians, from +z). |
| `phi_inc` | `Float64` | Azimuthal angle of incidence (radians, from +x). |
| `pol` | `Vec3` | Polarization unit vector. |
| `weight` | `Float64` | Weight in objective (default 1.0 if omitted). |

**Returns:** `Vector{AngleConfig}` of length `M`.

**Example:**

```julia
grid = make_sph_grid(36, 36)

angles = [
    (theta_inc=0.0,    phi_inc=0.0,   pol=Vec3(1,0,0), weight=1.0),   # broadside
    (theta_inc=pi/6,   phi_inc=0.0,   pol=Vec3(1,0,0), weight=1.0),   # 30 deg
    (theta_inc=pi/4,   phi_inc=0.0,   pol=Vec3(1,0,0), weight=1.0),   # 45 deg
    (theta_inc=pi/3,   phi_inc=0.0,   pol=Vec3(1,0,0), weight=1.0),   # 60 deg
]

configs = build_multiangle_configs(mesh, rwg, k, angles;
                                    grid=grid, backscatter_cone=10.0)
println("Configured $(length(configs)) angles")
```

---

## 3. The Optimizer

### `optimize_multiangle_rcs(Z_base, Mp, configs, theta0; kwargs...)`

Minimize total weighted backscatter RCS using projected L-BFGS with adjoint gradients. Supports any `AbstractMatrix{ComplexF64}` as the base operator.

**Required parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Z_base` | `AbstractMatrix{ComplexF64}` | Base EFIE operator (MLFMA, ACA, or dense). |
| `Mp` | `Vector{<:AbstractMatrix}` | Patch mass matrices from `precompute_patch_mass`. |
| `configs` | `Vector{AngleConfig}` | From `build_multiangle_configs`. |
| `theta0` | `Vector{Float64}` | Initial impedance parameter vector (length P). |

**Keyword arguments:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `maxiter` | `Int` | `100` | Maximum L-BFGS iterations. |
| `tol` | `Float64` | `1e-6` | Gradient-norm convergence tolerance. |
| `m_lbfgs` | `Int` | `10` | L-BFGS memory length. |
| `alpha0` | `Float64` | `0.01` | Initial inverse-Hessian scaling. |
| `verbose` | `Bool` | `true` | Print iteration progress. |
| `reactive` | `Bool` | `false` | `false` = resistive, `true` = reactive loading. |
| `lb` | `Vector` or `nothing` | `nothing` | Lower bounds on theta. |
| `ub` | `Vector` or `nothing` | `nothing` | Upper bounds on theta. |
| `preconditioner` | `AbstractPreconditionerData` or `nothing` | `nothing` | GMRES preconditioner (strongly recommended for convergence). |
| `gmres_tol` | `Float64` | `1e-6` | GMRES relative tolerance. |
| `gmres_maxiter` | `Int` | `300` | Maximum GMRES iterations per solve. |

**Returns:** Tuple `(theta_opt, trace)` where:
- `theta_opt::Vector{Float64}`: Optimized impedance parameters.
- `trace::Vector{NamedTuple}`: Records with fields `(iter, J, gnorm)`.

**Note:** The optimizer always uses GMRES internally. The `ImpedanceLoadedOperator` is matrix-free and does not support direct (LU) solves. A near-field preconditioner is strongly recommended.

---

## 4. Per-Iteration Algorithm

Each iteration of `optimize_multiangle_rcs` performs:

```
1. Build composite operator:    Z_op = ImpedanceLoadedOperator(Z_base, Mp, theta, reactive)
2. Forward solves (M angles):   I_a = Z_op \ v_a           (GMRES)
3. Evaluate objective:          J = sum_a w_a * Re(I_a' Q_a I_a)
4. Adjoint solves (M angles):   Z_op' lambda_a = Q_a I_a   (GMRES)
5. Accumulate gradient:         g[p] = sum_a w_a * gradient_impedance(Mp, I_a, lambda_a)
6. L-BFGS two-loop recursion:   compute search direction d
7. Projected line search:       backtracking Armijo with M forward solves per trial
8. Update theta:                theta = project(theta + alpha * d)
```

---

## 5. Complete Example: Multi-Angle RCS Reduction

### 5.1 Small Problem (Dense Operator)

```julia
using DiffMoM

# Geometry
lambda = 1.0
k = 2pi / lambda
mesh = make_rect_plate(2*lambda, 2*lambda, 20, 20)
rwg = build_rwg(mesh)
N = rwg.nedges

# Spatial patch assignment
partition = assign_patches_grid(mesh; nx=4, ny=4)
Mp = precompute_patch_mass(mesh, rwg, partition)
P = partition.P

# Base EFIE operator (dense for small problem)
Z_efie = assemble_Z_efie(mesh, rwg, k)

# Multi-angle configuration: 2 incidence angles
grid = make_sph_grid(36, 36)
angles = [
    (theta_inc=0.0,   phi_inc=0.0, pol=Vec3(1,0,0), weight=1.0),
    (theta_inc=pi/4,  phi_inc=0.0, pol=Vec3(1,0,0), weight=1.0),
]
configs = build_multiangle_configs(mesh, rwg, k, angles;
                                    grid=grid, backscatter_cone=10.0)

# Preconditioner
P_nf = build_nearfield_preconditioner(Z_efie, mesh, rwg, 1.0*lambda)

# Optimize
theta0 = zeros(P)
theta_opt, trace = optimize_multiangle_rcs(
    Z_efie, Mp, configs, theta0;
    lb=zeros(P), ub=fill(500.0, P),
    preconditioner=P_nf,
    maxiter=50, verbose=true
)

println("Final J: ", trace[end].J)
println("Objective reduction: ", round(trace[end].J / trace[1].J * 100, digits=1), "%")
```

### 5.2 Large Problem (MLFMA Operator)

```julia
using DiffMoM

# Large geometry
lambda = 1.0
k = 2pi / lambda
mesh = read_obj_mesh("aircraft.obj")
rwg = build_rwg(mesh)
N = rwg.nedges   # e.g., N = 15,000

# Spatial patches: coat nose cone and leading edges
regions = [
    region_sphere(center=Vec3(2.0, 0.0, 0.0), radius=0.3),  # nose cone
    region_box(lo=Vec3(-1.0, -2.0, -0.05),
               hi=Vec3(0.5, 2.0, 0.05)),                      # wing LE
]
partition = assign_patches_by_region(mesh, regions)
Mp = precompute_patch_mass(mesh, rwg, partition)

# MLFMA operator
mlfma = build_mlfma_operator(mesh, rwg, k; leaf_lambda=1.0)

# Preconditioner (built from MLFMA near-field blocks)
P_nf = build_nearfield_preconditioner(mlfma, 1.0*lambda)

# Multi-angle: 4 angles around the nose
grid = make_sph_grid(36, 36)
angles = [
    (theta_inc=0.0,    phi_inc=0.0,   pol=Vec3(0,1,0), weight=1.0),
    (theta_inc=pi/12,  phi_inc=0.0,   pol=Vec3(0,1,0), weight=1.0),
    (theta_inc=pi/12,  phi_inc=pi/2,  pol=Vec3(1,0,0), weight=1.0),
    (theta_inc=pi/6,   phi_inc=0.0,   pol=Vec3(0,1,0), weight=1.0),
]
configs = build_multiangle_configs(mesh, rwg, k, angles;
                                    grid=grid, backscatter_cone=10.0)

# Optimize
theta0 = zeros(partition.P)
lb = zeros(partition.P)
ub = fill(500.0, partition.P)
# Keep background patch (PEC) fixed
lb[end] = 0.0; ub[end] = 0.0   # last patch = background

theta_opt, trace = optimize_multiangle_rcs(
    mlfma, Mp, configs, theta0;
    lb=lb, ub=ub,
    preconditioner=P_nf,
    maxiter=100, gmres_tol=1e-6, gmres_maxiter=300,
    verbose=true
)
```

---

## 6. Design Considerations

### 6.1 Choosing Incidence Angles

- **Uniform coverage:** Distribute angles uniformly over the expected threat sector (e.g., forward hemisphere for aircraft).
- **Weighting:** Assign higher weights to more important angles (e.g., nose-on for aircraft).
- **Symmetry:** Exploit geometric symmetry to reduce the number of angles (e.g., for a body of revolution, one phi angle suffices).

### 6.2 Backscatter Cone Size

The `backscatter_cone` parameter controls the angular resolution of the RCS objective:

| Cone half-angle | Effect | Use case |
|-----------------|--------|----------|
| 5 deg | Tight, precise backscatter | Fine grids, narrow-beam applications |
| 10 deg (default) | Good balance | Most applications |
| 20 deg | Broad, robust | Coarse grids, initial exploration |

Larger cones are more robust to grid discretization but capture power from neighboring directions.

### 6.3 Number of Angles vs. Convergence

More angles increase per-iteration cost linearly but improve the quality of the multi-angle objective. Typical guidelines:

| Problem type | Recommended M | Notes |
|-------------|---------------|-------|
| Single-sector RCS | 2--4 | Cover the primary threat sector |
| Hemisphere coverage | 6--12 | Uniform spacing over forward hemisphere |
| Full-sphere coverage | 12--26 | Icosahedral or uniform grid on sphere |

### 6.4 Passive vs. Reactive Impedance

- **Resistive (`reactive=false`):** Absorptive coatings. Physical, always passive for `theta >= 0`. Use `lb=0` for passivity.
- **Reactive (`reactive=true`):** Lossless frequency-selective surfaces. Can redirect energy but not absorb it. Useful for RCS shaping rather than reduction.

For pure RCS reduction, resistive loading with passivity constraints (`lb=0`) is the standard approach.

---

## 7. Post-Optimization Validation

### 7.1 Recompute RCS at Each Angle

After optimization, independently verify the RCS at each incidence angle:

```julia
Z_opt = ImpedanceLoadedOperator(Z_base, Mp, theta_opt)
for (a, cfg) in enumerate(configs)
    I_a = solve_forward(Z_opt, cfg.v; solver=:gmres, preconditioner=P_nf)
    E_ff = compute_farfield(G_mat, I_a, length(grid.w))
    rcs_a = backscatter_rcs(E_ff, grid, cfg.k_vec / norm(cfg.k_vec))
    println("Angle $a: backscatter RCS = $(round(10*log10(rcs_a.sigma), digits=1)) dBsm")
end
```

### 7.2 Check Angles Not in the Objective

Verify that RCS hasn't increased at angles not included in the optimization:

```julia
# Test at intermediate angles
test_angles = [(theta_inc=pi/8, phi_inc=0.0, pol=Vec3(1,0,0), weight=1.0)]
test_configs = build_multiangle_configs(mesh, rwg, k, test_angles; grid=grid)
```

---

## 8. Code Mapping

| File | Contents |
|------|----------|
| `src/optimization/MultiAngleRCS.jl` | `AngleConfig`, `build_multiangle_configs`, `optimize_multiangle_rcs` |
| `src/assembly/CompositeOperator.jl` | `ImpedanceLoadedOperator` (used internally by optimizer) |
| `src/assembly/SpatialPatches.jl` | Spatial patch assignment utilities |
| `src/optimization/QMatrix.jl` | `direction_mask`, `build_Q`, `cap_mask` |
| `src/optimization/Adjoint.jl` | `solve_adjoint_rhs`, `gradient_impedance` |

---

## 9. Exercises

### 9.1 Conceptual Questions

1. **Angle interaction:** Why might optimizing for two incidence angles at 0 and 90 degrees give a better overall design than optimizing for just 0 degrees, even if we only care about 0-degree RCS?
2. **Weight selection:** How would you choose weights $w_a$ for an aircraft where nose-on RCS is 3x more important than broadside RCS?
3. **Cost scaling:** If a single-angle optimization takes 10 seconds per iteration with N = 5000, approximately how long would a 4-angle optimization take per iteration? What dominates the cost?

### 9.2 Coding Exercises

1. **Two-angle plate:** Optimize a 2-lambda plate for simultaneous RCS reduction at 0 and 30 degrees incidence. Compare the result against single-angle optimization at each angle separately.
2. **Spatial patches:** Use `assign_patches_by_region` to coat only one half of a plate, keeping the other half PEC. Optimize for broadside RCS and discuss the result.
3. **MLFMA integration:** For a sphere with N ~ 2000, compare optimization with a dense base operator vs. an MLFMA base operator. Verify that both converge to similar solutions.

### 9.3 Advanced Challenges

1. **Adaptive weighting:** Implement a scheme that adjusts $w_a$ each iteration based on the per-angle RCS values (give higher weight to the angle with highest remaining RCS). Does this improve convergence?
2. **Minimax objective:** Modify the optimization to minimize $\max_a \sigma_a$ instead of $\sum_a w_a \sigma_a$. Hint: use a smooth approximation like log-sum-exp.

---

## 10. Chapter Checklist

After studying this chapter, you should be able to:

- [ ] **Explain** why multi-angle optimization is necessary for practical RCS reduction.
- [ ] **Configure** multi-angle problems using `build_multiangle_configs` with appropriate angles, weights, and cone sizes.
- [ ] **Run** `optimize_multiangle_rcs` with dense, ACA, or MLFMA base operators.
- [ ] **Use** `ImpedanceLoadedOperator` to wrap any base operator with impedance loading.
- [ ] **Use** `direction_mask` to define backscatter cones around arbitrary directions.
- [ ] **Validate** multi-angle results by recomputing RCS at optimized and test angles.
- [ ] **Select** appropriate spatial patch strategies for complex geometries.

---

## 11. Further Reading

1. **Multi-angle RCS optimization:**
   - Kazemzadeh, A. (2011). *Nonmagnetic ultrawideband absorber with optimal thickness*. IEEE Transactions on Antennas and Propagation, 59(1), 135--140.
   - Gustafsson, M., et al. (2012). *Physical limitations on antennas of arbitrary shape*. Proceedings of the Royal Society A, 468(2139), 1--23.

2. **Adjoint methods for electromagnetic design:**
   - Nikolova, N. K., et al. (2004). *Adjoint techniques for sensitivity analysis in high-frequency structure CAD*. IEEE Transactions on Microwave Theory and Techniques, 52(1), 403--419.
   - Hassan, E., et al. (2014). *Topology optimization of metallic antennas*. IEEE Transactions on Antennas and Propagation, 62(5), 2488--2500.

3. **Fast multipole method:**
   - Chew, W. C., et al. (2001). *Fast and efficient algorithms in computational electromagnetics*. Artech House.
   - Ergul, O., & Gurel, L. (2014). *The multilevel fast multipole algorithm (MLFMA) for solving large-scale computational electromagnetics problems*. Wiley-IEEE Press.

---

*This chapter completes Part III -- Differentiable Design, extending the framework from single-objective optimization (Chapters 1--4) to the multi-angle, fast-operator setting needed for realistic RCS reduction design.*
