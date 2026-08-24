# Tutorial 3: Beam-steering design

This tutorial optimizes a reactive impedance pattern on a PEC plate. The
objective is the fraction of one projected far-field component that falls inside
a target cone.

Run the gradient checks in [Tutorial 2](02-adjoint-gradient-check.md) before
using the optimizer on a new formulation.

## Run the reference workflow

From the repository root:

```bash
julia --project=. validation/paper/run_beam_steering_case.jl
```

The driver writes these files under `data/`:

- `beam_steer_trace.csv` for accepted optimizer iterates;
- `beam_steer_impedance.csv` for triangle coordinates and mapped impedance;
- `beam_steer_impedance_cells.csv` for the independent design variables;
- `beam_steer_farfield.csv` for full-grid PEC and optimized directivity; and
- `beam_steer_cut_phi0.csv` for the sampled cut nearest $\phi=0$.

The reported objective, gradient, power ratio, and conditioning values come from
that invocation. Do not replace them with a fixed documentation transcript.

The numeric gates are the `BEAM_MAX_*` constants in
`validation/paper/run_beam_steering_case.jl`; the `checks` block applies those
same values and prints them with each result. Before writing the files, the
driver requires:

- finite scalar metrics, design variables, gradient errors, and trace entries;
- bounded PEC and optimized relative residuals;
- bounded PEC and optimized power-ratio errors;
- an optimized objective greater than the PEC objective;
- a bounded maximum symmetric gradient error; and
- design variables within the declared box bounds.

## Objective

Let $\mathbf I(\boldsymbol\theta)$ solve the impedance-loaded EFIE. The driver
maximizes

```math
J(\boldsymbol\theta)=
\frac{\mathbf I^\dagger \mathbf Q_{\mathrm{target}}\mathbf I}
     {\mathbf I^\dagger \mathbf Q_{\mathrm{total}}\mathbf I}.
```

`Q_target` sums the weighted, projected radiation-vector products within the
target cone. `Q_total` uses the same grid and polarization projection over the
full sphere. The ratio is therefore tied to that selected projection; it is not
the total electromagnetic directivity unless the chosen projection represents
the desired total field quantity.

The quotient rule requires one adjoint contribution for each quadratic form:

```math
\nabla J =
\frac{P_{\mathrm{total}}\nabla P_{\mathrm{target}}
      -P_{\mathrm{target}}\nabla P_{\mathrm{total}}}
     {P_{\mathrm{total}}^2}.
```

The example uses a purely reactive sheet impedance, $Z_s=i\theta$, with real
design variables. A separate unit-cell or material model is needed to map those
sheet-reactance values to a fabricable structure.

## Assemble the fixed problem

The following bounded setup reproduces the same kind of plate problem manually.
The `freq`, `Nx`, and `Ny` assignments in
`validation/paper/run_beam_steering_case.jl` own the driver's executable setup:

```julia
using DiffMoM
using LinearAlgebra

frequency = 3.0e9
c0 = 299792458.0
wavelength = c0 / frequency
k = 2π / wavelength
eta0 = 376.730313668

Nx = Ny = 12
Lx = Ly = wavelength
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg = build_rwg(mesh)
Nt = ntriangles(mesh)

Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
k_vec = Vec3(0.0, 0.0, -k)
v = assemble_v_plane_wave(
    mesh, rwg, k_vec, 1.0, Vec3(1.0, 0.0, 0.0);
    quad_order=3,
)
```

The design has one variable per rectangular cell. The following mapping assigns
both triangles in a cell to the same patch:

```julia
tri_cell = zeros(Int, Nt)
for t in 1:Nt
    center = triangle_center(mesh, t)
    ix = clamp(floor(Int, (center[1] + Lx / 2) / (Lx / Nx)) + 1, 1, Nx)
    iy = clamp(floor(Int, (center[2] + Ly / 2) / (Ly / Ny)) + 1, 1, Ny)
    tri_cell[t] = (iy - 1) * Nx + ix
end

partition = PatchPartition(tri_cell, Nx * Ny)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
```

## Build the target and total Q matrices

The target is a cone of half-angle $5^\circ$ centered at
$(\theta,\phi)=(30^\circ,0^\circ)$. `make_sph_grid(180, 72)` creates midpoint
samples with $1^\circ$ polar spacing and $5^\circ$ azimuth spacing.

```julia
grid = make_sph_grid(180, 72)
G = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
polarization = pol_linear_x(grid)

theta_target = deg2rad(30.0)
phi_target = 0.0
r_target = Vec3(
    sin(theta_target) * cos(phi_target),
    sin(theta_target) * sin(phi_target),
    cos(theta_target),
)

mask = BitVector([
    acos(clamp(dot(Vec3(grid.rhat[:, q]), r_target), -1.0, 1.0)) <=
        deg2rad(5.0)
    for q in eachindex(grid.w)
])

Q_target = build_Q(G, grid, polarization; mask=mask)
Q_total = build_Q(G, grid, polarization)
```

Keep the radiation operator, grid, polarization matrix, and mask from the same
setup. Mixing any of them invalidates the objective comparison.

## Establish the PEC reference

```julia
I_pec = solve_forward(Z_efie, v; solver=:direct)
P_target_pec = real(dot(I_pec, Q_target * I_pec))
P_total_pec = real(dot(I_pec, Q_total * I_pec))
J_pec = P_target_pec / P_total_pec
```

Record the original-system residual and verify mesh and angular-grid convergence
before using this value as a reference.

## Initialize and optimize

The reference driver starts from an x-directed reactance ramp and applies an
example box bound of $500\ \Omega$. That bound belongs to this numerical study;
it is not a general fabrication limit.

```julia
cell_x = [
    -Lx / 2 + (((p - 1) % Nx) + 0.5) * (Lx / Nx)
    for p in 1:(Nx * Ny)
]
x_halfspan = Lx / 2 - Lx / (2Nx)
theta_initial = 300.0 .* cell_x ./ x_halfspan

theta_opt, trace = optimize_directivity(
    Z_efie,
    Mp,
    v,
    Q_target,
    Q_total,
    theta_initial;
    reactive=true,
    maxiter=300,
    tol=1e-12,
    lb=-500.0,
    ub=500.0,
    alpha0=1e8,
    verbose=true,
)
```

`trace` contains accepted iterates. Report its objective and gradient norm, then
count active bounds from `theta_opt`. A flat trace alone does not identify the
cause of a stalled run.

## Recompute and verify the result

Rebuild the final system and objective outside the optimizer:

```julia
Z_opt = assemble_full_Z(Z_efie, Mp, theta_opt; reactive=true)
I_opt = solve_forward(Z_opt, v; solver=:direct)

P_target_opt = real(dot(I_opt, Q_target * I_opt))
P_total_opt = real(dot(I_opt, Q_total * I_opt))
J_opt = P_target_opt / P_total_opt

println("PEC objective = ", J_pec)
println("optimized objective = ", J_opt)
println("relative residual = ", norm(Z_opt * I_opt - v) / norm(v))
```

Verify several unconstrained gradient components at the final point:

```julia
lambda_target = Z_opt' \ (Q_target * I_opt)
lambda_total = Z_opt' \ (Q_total * I_opt)
gradient_target = gradient_impedance(
    Mp, I_opt, lambda_target; reactive=true)
gradient_total = gradient_impedance(
    Mp, I_opt, lambda_total; reactive=true)
gradient_adjoint =
    (P_total_opt .* gradient_target .-
     P_target_opt .* gradient_total) ./ P_total_opt^2

function objective(theta)
    Z = assemble_full_Z(Z_efie, Mp, theta; reactive=true)
    current = Z \ v
    numerator = real(dot(current, Q_target * current))
    denominator = real(dot(current, Q_total * current))
    return numerator / denominator
end

for p in 1:min(5, length(theta_opt))
    gradient_fd = fd_grad(objective, theta_opt, p; h=1e-5)
    scale = max(abs(gradient_adjoint[p]), abs(gradient_fd))
    relative_error = iszero(scale) ? 0.0 :
        abs(gradient_adjoint[p] - gradient_fd) / scale
    println((parameter=p, adjoint=gradient_adjoint[p],
             finite_difference=gradient_fd, relative_error))
end
```

Repeat the finite-difference check over a step-size sweep. At an active box
bound, use the projected gradient for the optimizer stationarity check; the
unconstrained component need not vanish.

## Interpret a run

Use measured checks instead of visual labels:

1. Confirm all forward and adjoint solves meet the declared true-residual gate.
2. Compare `J_opt` with `J_pec` using the same Q matrices.
3. Refine the angular grid before claiming a peak direction or sidelobe level.
4. Refine the mesh and compare the reported objective and pattern features.
5. Count active bounds and evaluate the projected gradient.
6. Verify selected gradient components over multiple finite-difference steps.

If the line search rejects every trial, evaluate the objective and directional
derivative at the accepted point, then sweep `alpha0` in both directions. If a
gradient check fails only at the final design, record the step sweep, solve
residuals, active bounds, and conditioning. Do not add regularization or change
the finite-difference step until a controlled test shows which factor changes
the mismatch.

## Code map

| Task | Source |
|:--|:--|
| Reference driver | `validation/paper/run_beam_steering_case.jl` |
| Directivity-ratio optimizer | `src/optimization/Optimize.jl` |
| Q matrices and polarization projection | `src/optimization/QMatrix.jl` |
| Adjoint solve and impedance gradient | `src/optimization/Adjoint.jl` |
| Finite-difference verification | `src/optimization/Verification.jl` |
| Far-field operators | `src/postprocessing/FarField.jl` |

For a quicker interactive case with a coarser angular grid, see
`examples/03_beamsteering_physical_unitcell.jl`. Use the paper driver when you
need its CSV schema or inputs.
