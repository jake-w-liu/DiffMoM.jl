# PEC sphere versus Mie reference

This benchmark compares the complete PEC EFIE, solve, far-field, and RCS path
with the package's analytical Mie-series implementation. It checks scaling,
polarization, and propagation and observation directions on a closed sphere.

It does not validate local fields. Use the
[Rayleigh-sphere near and total field check](06-near-total-field-rayleigh-sphere.md)
for `compute_nearfield` and `compute_total_field`.

## Analytical quantity

`mie_s1s2_pec(x, mu)` evaluates the two PEC-sphere scattering amplitudes for
$x=ka$ and $\mu=\cos\gamma$. The PEC coefficients are

```math
a_n=-\frac{[xj_n(x)]'}{[xh_n^{(2)}(x)]'},
\qquad
b_n=-\frac{j_n(x)}{h_n^{(2)}(x)}.
```

`mie_bistatic_rcs_pec(k, a, k_inc_hat, pol, rhat)` combines those amplitudes
with the incident polarization and returns

```math
\sigma(\hat{\mathbf r})=
\frac{4\pi}{k^2}
\left(|S_1c_\perp|^2+|S_2c_\parallel|^2\right).
```

With automatic truncation, the implemented order is

```math
n_{\max}=\max\left(3,\left\lceil
x+4x^{1/3}+2
\right\rceil\right).
```

The implementation validates the sphere size, directions, polarization, order,
and work limits and uses bounded high-precision fallbacks for exceptional
floating-point cases.

## Run the fixed example

```bash
julia --project=. --startup-file=no examples/04_pec_sphere_mie.jl
```

The example creates an icosphere, solves the dense EFIE, evaluates a midpoint
spherical grid, and compares the stored azimuth cut nearest zero with Mie values
at the same directions. Its `subdiv`, `a`, `freq`, and `grid` assignments own
the executable setup.

It exits nonzero unless the declared energy-ratio, cut-error, and backscatter
gates pass. The `MIE_EXAMPLE_MAX_*` constants in
`examples/04_pec_sphere_mie.jl` own the executable limits. The script builds
its error messages from those values.

The script prints measurements from the run and writes no artifacts. Preserve
its output and exit status instead of copying a fixed transcript into a report.

## Run the detailed validator

```bash
julia --project=. --startup-file=no validation/mie/validate_mie_rcs.jl
```

This driver generates an icosphere, writes and rereads its STL representation,
runs the MoM and Mie comparison, and applies gates to both stored E-plane and
H-plane cuts. It writes:

- `validation/mie/mie_rcs_phi0.csv`;
- `validation/mie/mie_rcs_phi90.csv`;
- `validation/mie/mie_rcs_summary.csv`; and
- four PNG files under `validation/mie/figs/`.

The tracked `validation/mie/sphere_ka2.1.stl` fixture can also be used for a
manual run.

## Manual comparison

Use the same stored directions on both paths:

```julia
using DiffMoM
using LinearAlgebra
using Statistics

frequency = 2.0e9
c0 = 299792458.0
k = 2π * frequency / c0
radius = 0.05

mesh = read_stl_mesh("validation/mie/sphere_ka2.1.stl")
rwg = build_rwg(mesh; allow_boundary=false, require_closed=true)

k_vec = Vec3(0.0, 0.0, -k)
k_inc_hat = k_vec / norm(k_vec)
polarization = Vec3(1.0, 0.0, 0.0)
wave = make_plane_wave(k_vec, 1.0, polarization)

Z = assemble_Z_efie(mesh, rwg, k; quad_order=3)
v = assemble_excitation(mesh, rwg, wave; quad_order=3)
current = solve_forward(Z, v; solver=:direct)

grid = make_sph_grid(60, 36)
G = radiation_vectors(mesh, rwg, grid, k; quad_order=3)
E_ff = compute_farfield(G, current, length(grid.w))
sigma_mom = bistatic_rcs(E_ff; E0=wave.E0)

phi_cut = minimum(grid.phi)
cut = findall(phi -> abs(phi - phi_cut) < 1e-12, grid.phi)
sort!(cut; by=index -> grid.theta[index])

sigma_mie = [
    mie_bistatic_rcs_pec(
        k,
        radius,
        k_inc_hat,
        polarization,
        Vec3(grid.rhat[:, index]),
    )
    for index in cut
]

floor_m2 = 1e-30
mom_db = 10 .* log10.(max.(sigma_mom[cut], floor_m2))
mie_db = 10 .* log10.(max.(sigma_mie, floor_m2))
delta_db = mom_db .- mie_db

metrics = (
    mae_db=mean(abs.(delta_db)),
    rmse_db=sqrt(mean(abs2, delta_db)),
    maximum_db=maximum(abs.(delta_db)),
)
println(metrics)
```

`make_sph_grid` uses midpoint samples, so the first stored azimuth is not
exactly zero. Label a cut with its recorded angle.

## Backscatter direction

`backscatter_rcs` accepts the incident propagation direction and selects the
stored sample nearest its negative:

```julia
back_mom = backscatter_rcs(
    E_ff, grid, k_inc_hat; E0=wave.E0)
back_mie = mie_bistatic_rcs_pec(
    k, radius, k_inc_hat, polarization, -k_inc_hat)

back_error_db =
    10log10(max(back_mom.sigma, floor_m2)) -
    10log10(max(back_mie, floor_m2))

println((
    error_db=back_error_db,
    angular_error_deg=back_mom.angular_error_deg,
))
```

For propagation along negative z, forward scattering is toward negative z and
backscatter is toward positive z.

## Radius of an imported sphere

For a mesh whose center is not already known, a vertex-radius summary is a
diagnostic, not a proof that the surface is spherical:

```julia
center = Vec3(vec(mean(mesh.xyz; dims=2)))
radii = [
    norm(Vec3(mesh.xyz[:, index]) - center)
    for index in 1:nvertices(mesh)
]
radius_estimate = mean(radii)
radius_standard_deviation = std(radii)
```

Record how the center and effective radius were chosen. Compare the result with
the intended CAD definition before using it in the Mie reference.

## Convergence study

For a changed electrical size or mesh:

1. record triangle count and a measured edge-size statistic;
2. verify the closed-surface mesh policy;
3. record the original-system true residual;
4. refine the spherical grid with the current held fixed;
5. refine the sphere mesh and reassemble;
6. compare supported triangle quadrature orders; and
7. inspect both linear-scale and dB errors near nulls.

Fit a convergence rate only if consecutive data show an asymptotic interval.
The fixed example gates belong to its geometry and frequency; they are not a
universal Mie-validation tolerance.

## Diagnose disagreement

| Pattern | First checks |
|:--|:--|
| Nearly constant dB offset | Incident amplitude, RCS normalization, radius, and units |
| Angular shift | Propagation direction, observation direction, and stored midpoint angles |
| Large dB spikes near nulls | Linear-scale error and the declared dB floor |
| E-plane and H-plane asymmetry | Mesh geometry, winding, polarization, and matched cuts |
| No refinement trend | Mesh quality, resolution, quadrature, residual, and Mie inputs |

A pattern suggests a controlled comparison; it does not establish the cause.

## Code map

| Task | Source |
|:--|:--|
| PEC and dielectric Mie functions | `src/postprocessing/Mie.jl` |
| Fixed PEC sphere example | `examples/04_pec_sphere_mie.jl` |
| Detailed STL and cut validator | `validation/mie/validate_mie_rcs.jl` |
| RCS and backscatter diagnostics | `src/postprocessing/Diagnostics.jl` |
| Far-field construction | `src/postprocessing/FarField.jl` |
