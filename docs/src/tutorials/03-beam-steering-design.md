# Tutorial: Beam Steering Design

## Purpose

Beam steering is the canonical benchmark for differentiable metasurface design. This tutorial walks through the complete inverse‑design workflow:

- **Formulate a directivity‑ratio objective** that naturally steers radiation without artificial penalties
- **Configure target and total `Q` matrices** that define “power in a cone” vs “total radiated power”
- **Run projected L‑BFGS optimization** with reactive impedance ($Z_s = i\theta$) for lossless phase control
- **Analyze convergence traces and far‑field patterns** to verify steering performance

By the end, you will have optimized a 4λ × 4λ reactive impedance sheet to steer a normally‑incident plane wave to $30^\circ$ off broadside, achieving a 5–10× directivity improvement over a PEC plate.

**Prerequisite:** Complete **Tutorial 2: Adjoint Gradient Check** to ensure your gradient implementation is correct before attempting optimization.

---

## Learning Goals

After this tutorial, you should be able to:

1. **Understand the directivity‑ratio objective** $J = P_\text{target}/P_\text{total}$ and why it avoids trivial solutions.
2. **Build `Q_target` and `Q_total` matrices** for arbitrary angular cones using `build_Q` and spherical‑grid masks.
3. **Initialize impedance parameters** with a physical phase ramp that jump‑starts optimization.
4. **Run `optimize_directivity`** with box constraints, preconditioning, and convergence monitoring.
5. **Interpret optimization traces** (objective, gradient norm, step size) to diagnose convergence.
6. **Evaluate steering performance** through far‑field cuts, directivity metrics, and pattern comparisons.
7. **Troubleshoot common pitfalls**: stalled optimization, poor initialization, constraint violations.

---

## Mathematical Background

### Directivity Ratio Objective

A naive objective $P_\text{target} = I^\dagger Q_\text{target} I$ can be trivially maximized by simply increasing the total radiated power, which broadens the beam rather than steering it. The **directivity ratio**

\[
J(\theta) = \frac{P_\text{target}(\theta)}{P_\text{total}(\theta)}
          = \frac{I(\theta)^\dagger Q_\text{target} I(\theta)}
                 {I(\theta)^\dagger Q_\text{total}  I(\theta)},
\]

where $Q_\text{total}$ corresponds to integration over the full sphere, measures the **fraction** of radiated power that lands in the target cone. Maximising this fraction naturally steers energy toward the desired direction while suppressing sidelobes.

The gradient of this ratio requires **two adjoint solves** (paper Eq. 30):

\[
\nabla J = \frac{P_\text{total} \nabla P_\text{target} - P_\text{target} \nabla P_\text{total}}{P_\text{total}^2}.
\]

Both $\nabla P_\text{target}$ and $\nabla P_\text{total}$ are computed via the standard adjoint formula (paper Eq. 25). The function `optimize_directivity` (`src/optimization/Optimize.jl`) implements this ratio‑gradient efficiently.

### Reactive Impedance for Beam Steering

For lossless beam steering we use **purely reactive surface impedance** $Z_s = i\theta$ ($\theta \in \mathbb{R}$). The imaginary unit $i$ introduces a phase shift $\phi \approx \arg(1 + i\theta)$ across the aperture, creating the phase gradient needed to steer the beam. Reactive impedance conserves power ($\Re\{Z_s\}=0$), making it the physical mechanism for phase‑only metasurfaces.

### Angular Grid and Cone Mask

The far‑field is sampled on a spherical grid (`SphGrid`) with $N_\Omega$ directions $(\theta_q,\phi_q)$ and quadrature weights $w_q$. The **target cone** is defined as all directions within an angular radius $\Delta$ of the steering direction $(\theta_s,\phi_s)$:

\[
\text{mask}[q] = \bigl[\arccos(\hat{\mathbf{r}}_q \cdot \hat{\mathbf{r}}_s) \le \Delta\bigr].
\]

The matrix $Q_\text{target}$ is built by summing radiation‑vector outer products over the masked directions (paper Eq. 15). $Q_\text{total}$ sums over all directions.

---

## Step‑by‑Step Workflow

### 1) Problem Setup: Frequency, Geometry, Mesh

Start with a 4λ × 4λ plate at 3 GHz, discretised with $\sim \lambda/3$ triangle spacing for good spatial resolution.

```julia
using DiffMoM
using LinearAlgebra

# Frequency and wavelength
freq = 3e9
c0   = 299792458.0
lambda0 = c0 / freq
k    = 2π / lambda0
eta0 = 376.730313668

# Plate geometry: 4λ × 4λ aperture
Lx = 4 * lambda0
Ly = 4 * lambda0
Nx, Ny = 12, 12       # ~λ/3 element size
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges
Nt   = ntriangles(mesh)

println("Mesh: $Nt triangles, $N RWG basis functions")
println("Plate size: $(round(Lx*100, digits=1)) cm × $(round(Ly*100, digits=1)) cm")
```

### 2) EFIE Assembly and Impedance Partition

Assemble the EFIE matrix and pre‑compute patch mass matrices `Mp` (one patch per triangle).

```julia
Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)

partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
println("Impedance patches: $Nt")
```

### 3) Excitation: Normal‑Incidence Plane Wave

```julia
k_vec = Vec3(0.0, 0.0, -k)    # propagating -z
E0    = 1.0
pol_inc = Vec3(1.0, 0.0, 0.0) # x‑polarized
v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3)
```

### 4) Far‑Field Grid and Target Cone

Create a spherical grid with 1° resolution in $\theta$ and 5° in $\phi$, then define a $5^\circ$ cone around the steering direction $\theta_s = 30^\circ$, $\phi_s = 0$.

```julia
grid = make_sph_grid(180, 72)          # 180 θ points × 72 φ points
NΩ = length(grid.w)
println("Far‑field grid: $NΩ directions")

# Radiation vectors for all directions
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)

# Polarization matrix (θ‑hat for x‑polarized radiation)
pol_mat = pol_linear_x(grid)

# Steering direction
theta_steer = 30.0 * π / 180   # 30°
phi_steer   = 0.0
steer_rhat = Vec3(sin(theta_steer) * cos(phi_steer),
                  sin(theta_steer) * sin(phi_steer),
                  cos(theta_steer))

# Mask: directions within 5° of steer_rhat
mask = BitVector([begin
    rh = Vec3(grid.rhat[:, q])
    angle = acos(clamp(dot(rh, steer_rhat), -1.0, 1.0))
    angle <= 5.0 * π / 180
end for q in 1:NΩ])

n_target = count(mask)
println("Target cone (Δθ=5°): $n_target directions")
```

### 5) Build Q_target and Q_total

```julia
Q_target = build_Q(G_mat, grid, pol_mat; mask=mask)
Q_total  = build_Q(G_mat, grid, pol_mat)   # no mask → full sphere

# Sanity checks
println("Q_target Hermitian: $(norm(Q_target - Q_target') < 1e-12 * norm(Q_target))")
println("Q_total  Hermitian: $(norm(Q_total  - Q_total')  < 1e-12 * norm(Q_total))")
```

### 6) PEC Reference Solution

Compute the performance of a perfect electric conductor (no impedance sheet) as a baseline.

```julia
I_pec = Z_efie \ v
P_target_pec = real(dot(I_pec, Q_target * I_pec))
P_total_pec  = real(dot(I_pec, Q_total * I_pec))
J_pec = P_target_pec / P_total_pec
println("PEC directivity fraction: $(round(J_pec*100, digits=2))%")
```

### 7) Initialization: Physical Phase Ramp

A linear phase gradient across the aperture provides a physically sensible starting point. The required phase shift for steering to $\theta_s$ is $\phi(x) = k x \sin\theta_s$.

```julia
tri_centers = [triangle_center(mesh, t) for t in 1:Nt]
cx = [tc[1] for tc in tri_centers]
x_center = (minimum(cx) + maximum(cx)) / 2
x_halfspan = (maximum(cx) - minimum(cx)) / 2

# Impedance values proportional to x‑position (phase ramp)
theta_init = [300.0 * (cx_t - x_center) / x_halfspan for cx_t in cx]
println("Initial θ range: [$(round(minimum(theta_init), digits=1)), $(round(maximum(theta_init), digits=1))] Ω")
```

### 8) Run Optimization

Call `optimize_directivity` with box constraints $|\theta| \le 500\,\Omega$ (typical fabrication limits). Use reactive impedance (`reactive=true`).

```julia
theta_bound = 500.0
theta_opt, trace = optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, theta_init;
    reactive=true,
    maxiter=300,
    tol=1e-12,
    lb=fill(-theta_bound, Nt),
    ub=fill( theta_bound, Nt),
    alpha0=0.01,         # initial inverse Hessian scaling for L-BFGS
    verbose=true         # print iteration progress
)

println("Optimization completed after $(length(trace)) iterations")
println("Final objective J = $(trace[end].J)")
println("Final gradient norm = $(trace[end].gnorm)")
```

### 9) Post‑Optimization Analysis

#### 9.1 Compute Optimized Far‑Field

```julia
Z_opt = assemble_full_Z(Z_efie, Mp, theta_opt; reactive=true)
I_opt = Z_opt \ v
E_ff_opt = compute_farfield(G_mat, I_opt, NΩ)

P_target_opt = real(dot(I_opt, Q_target * I_opt))
P_total_opt  = real(dot(I_opt, Q_total * I_opt))
J_opt = P_target_opt / P_total_opt
improvement = J_opt / J_pec
println("Optimized directivity fraction: $(round(J_opt*100, digits=2))%")
println("Improvement over PEC: $(round(improvement, digits=2))×")
```

#### 9.2 Plot Convergence Trace

```julia
using Plots  # or PyPlot, GLMakie, etc.

iters = [t.iter for t in trace]
Js    = [t.J    for t in trace]
gnorms = [t.gnorm for t in trace]

plot(iters, Js, xlabel="Iteration", ylabel="J", label="Objective",
     title="Beam‑Steering Optimization Trace")
plot!(twinx(), iters, gnorms, color=:red, label="‖∇J‖", yscale=:log10)
```

#### 9.3 Extract Phi=0 Cut and Compare with PEC

```julia
# Compute PEC far field (needed for comparison)
E_ff_pec = compute_farfield(G_mat, I_pec, NΩ)

# Far‑field power pattern
ff_power_pec = [real(dot(E_ff_pec[:, q], E_ff_pec[:, q])) for q in 1:NΩ]
ff_power_opt = [real(dot(E_ff_opt[:, q], E_ff_opt[:, q])) for q in 1:NΩ]

# Directivity D = 4π|E|² / ∫|E|² dΩ
P_sphere_pec = sum(ff_power_pec[q] * grid.w[q] for q in 1:NΩ)
P_sphere_opt = sum(ff_power_opt[q] * grid.w[q] for q in 1:NΩ)
D_pec = [4π * ff_power_pec[q] / P_sphere_pec for q in 1:NΩ]
D_opt = [4π * ff_power_opt[q] / P_sphere_opt for q in 1:NΩ]

# Phi=0 cut (azimuthal plane containing steering direction)
dphi = 2π / 72
phi0_idx = [q for q in 1:NΩ if min(grid.phi[q], 2π - grid.phi[q]) <= dphi/2 + 1e-10]
phi0_sorted = sort(phi0_idx, by=q -> grid.theta[q])

theta_cut = grid.theta[phi0_sorted]
D_pec_cut = 10 .* log10.(max.(D_pec[phi0_sorted], 1e-30))
D_opt_cut = 10 .* log10.(max.(D_opt[phi0_sorted], 1e-30))

# Plot cut
plot(rad2deg.(theta_cut), D_pec_cut, label="PEC", xlabel="θ (deg)", ylabel="Directivity (dBi)")
plot!(rad2deg.(theta_cut), D_opt_cut, label="Optimized", lw=2)
vline!([rad2deg(theta_steer)], color=:black, ls=:dash, label="Target θ_s")
```

#### 9.4 Verify Gradient at Optimum

Re‑run gradient verification at the optimized point to ensure numerical consistency.

```julia
function J_of_theta_reactive(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= (1im * theta_vec[p]) .* Mp[p]
    end
    I_t = Z_t \ v
    f_t = real(dot(I_t, Q_target * I_t))
    g_t = real(dot(I_t, Q_total * I_t))
    return f_t / g_t
end

# Check first 5 parameters
for p in 1:5
    g_fd = fd_grad(J_of_theta_reactive, theta_opt, p; h=1e-5)
    # Compute adjoint gradient at optimum (two adjoint solves)
    lam_t = Z_opt' \ (Q_target * I_opt)
    lam_a = Z_opt' \ (Q_total * I_opt)
    g_f = gradient_impedance(Mp, I_opt, lam_t; reactive=true)
    g_g = gradient_impedance(Mp, I_opt, lam_a; reactive=true)
    g_adj = (P_total_opt .* g_f .- P_target_opt .* g_g) ./ (P_total_opt^2)
    rel_err = abs(g_adj[p] - g_fd) / max(abs(g_adj[p]), 1e-30)
    println("p=$p: adj=$(g_adj[p])  fd=$g_fd  rel_err=$rel_err")
end
```

---

## Interpretation Guidelines

### Convergence Trace

| Pattern | Likely Cause | Action |
|---------|--------------|--------|
| **Objective plateaus early** | Initial phase ramp insufficient; optimizer stuck near broadside | Increase ramp magnitude, add random perturbation |
| **Gradient norm oscillates** | Step size too large, ill‑conditioned Hessian | Reduce `alpha0` (inverse Hessian scaling), enable preconditioning |
| **Objective decreases then increases** | Line‑search fails due to gradient scaling | Scale objective (multiply by constant), check gradient sign |
| **Gradient norm stalls ~1e‑6** | Convergence reached; further improvement limited by discretisation | Accept solution, refine mesh if needed |

### Far‑Field Patterns

- **Peak directivity** should be within $1^\circ$ of target steering angle.
- **Sidelobe level** typically rises as target cone narrows (trade‑off between directivity and sidelobes).
- **Nulls** may shift; compare with PEC pattern to see how impedance sheet reshapes radiation.

### Impedance Distribution

Plot `theta_opt` vs $x$ position; it should approximate a linear phase ramp with superposed ripples that suppress sidelobes. Large discontinuities indicate numerical instability or insufficient regularization.

---

## Troubleshooting

### Optimization Stalls Near Broadside

**Symptom:** Objective improves only marginally, beam remains broadside.

**Causes and fixes:**

1. **Weak initial phase ramp** – Increase the coefficient in the linear ramp (e.g., from 300 to 600 Ω).
2. **Symmetrical initialization** – Ensure ramp is asymmetric (odd function) about the center; add a small random component.
3. **Gradient scaling** – Check that `alpha0` is not too large (it scales the initial inverse Hessian). Try `alpha0=0.001` or `alpha0=0.1`.
4. **Box constraints too tight** – If `theta_bound` is too small, the optimizer cannot create sufficient phase shift. Increase to 800–1000 Ω (if physically plausible).

### Line‑Search Failures (Step Size Repeatedly Reduced)

**Symptom:** Log shows “line search failed, reducing α” repeatedly.

**Solutions:**

- **Scale the objective** – Multiply $J$ by a constant (e.g., 100) to increase gradient magnitude.
- **Check gradient sign** – Verify gradient verification passes (Tutorial 2). A sign error causes line search to move uphill.
- **Increase `alpha0`** – Larger inverse Hessian scaling stabilises early steps.

### Poor Far‑Field Pattern (High Sidelobes, Split Beams)

**Causes:**

- **Target cone too narrow** – Broadening the cone (e.g., $10^\circ$) reduces sidelobes.
- **Insufficient spatial resolution** – Increase mesh density ($Nx, Ny$).
- **Reactive‑only limitation** – For wide‑angle steering, small resistive components may be needed; try `reactive=false` with small lower bound (e.g., `lb=1.0`).

### Gradient Verification Fails at Optimum

If verification passes at random `theta0` but fails at `theta_opt`:

- **Preconditioning inconsistency** – Ensure `preconditioner_M` is identical in forward, adjoint, and gradient.
- **Ill‑conditioned $Z(\theta_\text{opt})$** – Check condition number with `condition_diagnostics(Z_opt)`. Add regularization (`regularization_alpha=1e‑8`).
- **Finite‑difference step size** – Try `h=1e‑6` or `h=1e‑7`.

---

## Code Mapping

| Task | Function | Source File | Key Lines |
|------|----------|-------------|-----------|
| **Directivity‑ratio optimizer** | `optimize_directivity` | `src/optimization/Optimize.jl` | 50–150 |
| **Q‑matrix construction** | `build_Q` | `src/optimization/QMatrix.jl` | 60–90 |
| **Spherical grid** | `make_sph_grid` | `src/postprocessing/FarField.jl` | 30–50 |
| **Radiation vectors** | `radiation_vectors` | `src/postprocessing/FarField.jl` | 120–150 |
| **Polarization matrix** | `pol_linear_x` | `src/optimization/QMatrix.jl` | 130–150 |
| **Far‑field computation** | `compute_farfield` | `src/postprocessing/FarField.jl` | 200–220 |
| **Convergence tracing** | `Vector{NamedTuple{(:iter, :J, :gnorm)}}` | `src/optimization/Optimize.jl` | 20–30 |
| **Gradient verification** | `verify_gradient` | `src/optimization/Verification.jl` | 42–86 |

**Complete example:** `examples/03_beamsteering_physical_unitcell.jl` (297 lines) – includes the optimization workflow above. For packaged CSV artifacts and consistency inputs, run `validation/paper/run_beam_steering_case.jl`.

---

## Exercises

### Basic (45 minutes)

1. **Run the nominal steering case** with the default $5^\circ$ cone. Report final directivity fraction and improvement over PEC.
2. **Plot the convergence trace** (objective and gradient norm). Identify the iteration where convergence slows.
3. **Extract the phi=0 cut** and confirm the peak is within $1^\circ$ of $30^\circ$.

### Practical (90 minutes)

1. **Narrow the target cone** to $2^\circ$. How does the optimization change? Compare sidelobe levels and convergence speed.
2. **Switch to resistive impedance** (`reactive=false`). How does the optimized pattern differ? Explain the physical reason.
3. **Add left preconditioning** by passing `preconditioner_M=make_left_preconditioner(Mp)` to `optimize_directivity`. Does convergence improve? Verify gradients at optimum still match.

### Advanced (2 hours)

1. **Multi‑objective steering** – Create two target cones at $+30^\circ$ and $-30^\circ$ with equal weight. Build a combined $Q_\text{target}$ and optimize for dual‑beam generation.
2. **Frequency sweep** – Run the optimization at 2.5, 3.0, and 3.5 GHz (re‑assembling `Z_efie` each time). Plot directivity vs frequency to assess bandwidth.
3. **Shape‑impedance co‑design** – Perturb mesh vertices along $x$ (using `mesh.vertices .+= δ`) and compute finite‑difference shape derivatives. Compare with adjoint shape gradients from the paper.

---

## Tutorial Checklist

Before moving to more complex inverse‑design problems, ensure you have:

- [ ] **Successfully steered** a 4λ × 4λ aperture to $30^\circ$ with $>5\times$ directivity improvement.
- [ ] **Verified gradients** at the optimum (relative error $<10^{-6}$).
- [ ] **Plotted convergence traces** and identified the convergence rate.
- [ ] **Compared far‑field patterns** (PEC vs optimized) in both linear and dB scales.
- [ ] **Experimented with cone width** and observed the directivity–sidelobe trade‑off.
- [ ] **Saved impedance distribution** and phase‑ramp plot for documentation.

---

## Further Reading

- **Paper Section 4.2** – Beam‑steering case study with reactive impedance.
- **Paper Eq. 30** – Gradient of ratio objectives (two adjoint solves).
- **`src/optimization/Optimize.jl`** – Implementation of `optimize_directivity` with projected L‑BFGS.
- **`src/optimization/QMatrix.jl`** – Construction of $Q$ matrices for arbitrary angular masks.
- **Tutorial 4: Sphere‑Mie RCS** – Validation against analytical Mie series.
- **Tutorial 5: Airplane RCS** – Applying differentiable MoM to complex platforms.
