# Chapter 6: Near/Total-Field Rayleigh Sphere

## Purpose

Validate the local electric-field postprocessing routines
`compute_nearfield` and `compute_total_field` against an analytical reference:
the Rayleigh-limit response of an electrically small PEC sphere illuminated by a
plane wave. This chapter complements the sphere-vs-Mie benchmark:

- `validation/04-sphere-mie-benchmark.md` validates far-field and RCS output.
- This chapter validates scattered near-field and total electric field output.

The benchmark is based directly on `examples/21_near_total_field_rayleigh_sphere.jl`.

---

## Learning Goals

After this chapter, you should be able to:

1. Understand why the Rayleigh small-sphere limit is the right analytical benchmark for local electric fields.
2. Reproduce the benchmark for `compute_nearfield` and `compute_total_field`.
3. Interpret the expected error envelope for the MoM solution versus the analytical dipole-limit model.
4. Distinguish near-/total-field validation from far-field/RCS validation.
5. Use the example outputs and plots to diagnose local-field regressions.

---

## 1) Reference Model

### 1.1 Physical Regime

The benchmark uses an electrically small PEC sphere with

- radius `a = 0.02 m`
- frequency `f = 100 MHz`
- free-space wavelength `λ0 ≈ 2.998 m`
- size parameter `ka ≈ 0.0419`

This satisfies `ka << 1`, so the sphere response is well approximated by the
Rayleigh electric-dipole limit.

### 1.2 Analytical Fields

The incident field is an exact plane wave:

```math
\mathbf E^{\mathrm{inc}}(\mathbf r) = \mathbf p E_0 e^{-i \mathbf k \cdot \mathbf r}.
```

The scattered field is approximated by the electric field of an induced dipole
moment

```math
\mathbf p_{\mathrm{eq}} = 4 \pi \epsilon_0 a^3 E_0 \, \hat{\mathbf p}.
```

The example evaluates the dipole field in full vector form, including the
`1/R^3`, `1/R^2`, and `1/R` terms. The total field is then

```math
\mathbf E^{\mathrm{tot}}(\mathbf r) = \mathbf E^{\mathrm{inc}}(\mathbf r) + \mathbf E^{\mathrm{sca}}(\mathbf r).
```

### 1.3 Why This Benchmark, Not PO

Physical optics (PO) is a high-frequency approximation used for electrically
large scatterers and far-field-dominated workflows. It is not the right
reference for validating local electric fields around a small scatterer. The
Rayleigh dipole model is the correct analytical benchmark here because:

- it is derived for `ka << 1`,
- it gives a closed-form scattered electric field in space,
- it tests both `E_sca` and `E_tot` directly.

---

## 2) Benchmark Setup

The example script:

```bash
julia --project=. examples/21_near_total_field_rayleigh_sphere.jl
```

performs the following steps:

1. Builds a small icosphere mesh for a PEC sphere.
2. Solves the EFIE system under plane-wave illumination.
3. Evaluates `compute_nearfield` and `compute_total_field`.
4. Compares both to the analytical Rayleigh reference.
5. Saves validation data and plots.

### 2.1 Observation Points

Observation points are sampled along the `x` and `z` axes at

```math
r/a = 1.4:0.2:3.0.
```

This keeps the points off the surface while remaining in the near zone, where
local-field validation is meaningful.

### 2.2 API Pattern Used

The benchmark follows the recommended excitation-object workflow:

```julia
k_vec = Vec3(0.0, 0.0, -k)
pol = Vec3(1.0, 0.0, 0.0)
pw = make_plane_wave(k_vec, 1.0, pol)

v = assemble_excitation(mesh, rwg, pw; quad_order=3)
I = Z \ v

E_sca = compute_nearfield(mesh, rwg, I, obs_all, k; quad_order=7, eta0=eta0)
E_tot = compute_total_field(mesh, rwg, I, pw, obs_all, k; quad_order=7, eta0=eta0)
```

---

## 3) Output Artifacts

The script writes three useful artifacts:

- `data/rayleigh_near_total_field_validation.csv`
- `examples/figs/21_rayleigh_near_total_field_magnitude.png`
- `examples/figs/21_rayleigh_near_total_field_errors.png`

The docs do not depend on these files being pre-generated, but they are useful
for regression checks and reports after running the example locally.

---

## 4) Acceptance Envelope

The example enforces the following assertions:

- residual `< 1e-10`
- `max rel_err(E_sca) < 0.35`
- `max rel_err(E_tot) < 0.10`

These are the benchmark's acceptance criteria. Use them as the stable reference
for docs, CI, or local regression checks instead of copying one machine's exact
console output.

### 4.1 Why the Total-Field Error Is Lower

In this setup, `E_tot` is often less sensitive than `E_sca` because the exact
incident plane wave is added directly, while only the scattered component is
approximated by the Rayleigh dipole model. That means the dominant part of the
total field is known exactly, and the remaining model discrepancy is confined to
the scattered contribution.

---

## 5) How to Read the Results

The example reports relative errors along the `x` and `z` axes and saves plots
for:

- `|E_sca|` comparison: MoM versus Rayleigh reference
- `|E_tot|` comparison: MoM versus exact-incident-plus-Rayleigh reference
- relative error curves for both fields along both axes

Use the CSV when you want machine-readable diagnostics and the PNGs when you
want quick visual confirmation that the field trends and error envelope are
reasonable.

---

## 6) Common Interpretation Pitfalls

1. **Do not compare directly on the surface.**
   `compute_nearfield` and `compute_total_field` reject on-surface points by
   design, and the analytical dipole model is not intended for that limit.

2. **Do not treat this as a high-frequency benchmark.**
   If `ka` is not small, the Rayleigh approximation is no longer valid.

3. **Do not substitute PO for this validation target.**
   PO is useful elsewhere in the package, but not as the analytical reference
   for small-sphere local electric fields.

---

## 7) Related Pages

- `validation/01-internal-consistency.md` for solver self-consistency checks
- `validation/04-sphere-mie-benchmark.md` for exact far-field / RCS validation
- `api/farfield-rcs.md` for the `compute_nearfield` and `compute_total_field` API
- `examples/21_near_total_field_rayleigh_sphere.jl` for the full runnable script

---

## 8) Checklist

Before treating near-/total-field output as validated, confirm:

- [ ] The example runs to completion.
- [ ] The residual is below `1e-10`.
- [ ] `max rel_err(E_sca) < 0.35`.
- [ ] `max rel_err(E_tot) < 0.10`.
- [ ] The CSV and plots are generated as expected.

---

## 9) Further Reading

- `examples/21_near_total_field_rayleigh_sphere.jl`
- Balanis, *Advanced Engineering Electromagnetics* — dipole-field formulas
- Jackson, *Classical Electrodynamics* — small-particle / dipole scattering
