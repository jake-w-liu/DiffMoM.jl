# Singular Integration

## Purpose

This chapter documents the singular-integration paths used by `DiffMoM.jl` for EFIE self terms and adjacent-cell (edge-sharing) pairs. The goal is strict code-formulation correspondence to:

- `src/basis/Greens.jl`
- `src/assembly/SingularIntegrals.jl`
- `src/assembly/EFIE.jl`

Two classes of triangle pairs receive special treatment:

1. **Self-cell** (`tm == tn`): exact singularity extraction via analytical `∫ 1/R dS'`.
2. **Adjacent-cell** (edge-sharing pairs): singularity subtraction with analytical scalar-potential inner integral and high-order quadrature for the vector-potential inner integral.

---

## Learning Goals

After this chapter, you should be able to:

1. Explain why EFIE self terms and adjacent-cell terms need special treatment.
2. Derive the `G = G_smooth + 1/(4πR)` split used in code.
3. Describe how `analytical_integral_1overR` is used inside `self_cell_contribution` and `adjacent_cell_contribution`.
4. Map the self/adjacent/non-adjacent branching in `assemble_Z_efie` to the formulas.
5. Identify current limitations (what is not yet special-cased).

---

## 1. Where the Singularity Appears

In mixed-potential EFIE assembly, each entry uses integrals of the form

```math
\iint f_m(\mathbf r)\, f_n(\mathbf r')\, G(\mathbf r,\mathbf r')\, dS\, dS',
\qquad
G(\mathbf r,\mathbf r') = \frac{e^{-ikR}}{4\pi R},\; R=|\mathbf r-\mathbf r'|.
```

When source and test points approach each other (`R -> 0`), `1/R` is singular. For the EFIE implementation in this package, `_efie_entry` in `src/assembly/EFIE.jl` uses a three-way branch:

1. `tm == tn` (self-cell): `self_cell_contribution` with exact singularity extraction,
2. edge-sharing pairs (adjacent-cell): `adjacent_cell_contribution` with singularity subtraction and high-order inner quadrature,
3. all other pairs: standard product quadrature with the full Green's function.

---

## 2. Kernel Splitting Used in Code

The Green kernel is split as

```math
G(\mathbf r,\mathbf r')
=
\underbrace{\frac{e^{-ikR}-1}{4\pi R}}_{G_{\text{smooth}}}
+
\underbrace{\frac{1}{4\pi R}}_{G_{\text{sing}}}.
```

### 2.1 Smooth Part

`greens_smooth(r, rp, k)` computes

```math
G_{\text{smooth}} = \frac{e^{-ikR}-1}{4\pi R},
\qquad
\lim_{R\to 0} G_{\text{smooth}} = -\frac{i k}{4\pi}.
```

In code, the small-`R` limit is handled explicitly (`abs(R) < 1e-14`) and otherwise uses `expm1(-im*k*R)/(4πR)` for numerical stability.

### 2.2 Singular Part

The singular part is handled through

```math
S(\mathbf P) = \int_T \frac{1}{|\mathbf P-\mathbf r'|}\, dS',
```

computed by `analytical_integral_1overR(P, V1, V2, V3)`.

---

## 3. Analytical Inner Integral `analytical_integral_1overR`

`analytical_integral_1overR` evaluates the triangle integral using the Graglia (1993) / Wilton et al. (1984) formula, which works for both coplanar and off-plane observation points:

```math
S(\mathbf P) = \sum_{\text{edges }i} d_i
\log\!\left(\frac{s_i^+ + R_i^+}{s_i^- + R_i^-}\right)
\;-\; |h| \sum_{\text{edges }i}
\left[\arctan\!\frac{d_i\, s_i^+}{R_{0i}^2 + |h|\, R_i^+}
     - \arctan\!\frac{d_i\, s_i^-}{R_{0i}^2 + |h|\, R_i^-}\right]
```

where $h = (\mathbf P - \mathbf V_1)\cdot\hat n_T$ is the signed height of $\mathbf P$ above the triangle plane, $R_{0i}^2 = d_i^2 + h^2$, and all in-plane quantities ($d_i$, $s_i^\pm$) are computed relative to the projection $\boldsymbol\xi = \mathbf P - h\,\hat n_T$.

When $h = 0$ (coplanar), the arctan term vanishes and the formula reduces to the pure edge-log expression.

Practical behavior in code:

- degenerate triangle (`|n_T|` tiny): returns `0.0`,
- near-zero perpendicular distance to an edge (`|d_i| < 1e-15`): skips that edge's log and arctan contributions,
- near-zero log numerator/denominator: safely skipped,
- `|h| < 1e-15`: arctan term skipped (coplanar fast path).

This is implemented exactly in `src/assembly/SingularIntegrals.jl`.

---

## 4. Self-Cell Contribution Algorithm

`self_cell_contribution(...)` returns the self-triangle integral contribution for `(vec_part - scl_part)` before multiplying by `-iωμ0`.

It is decomposed into:

```math
I_{\text{self}} = I_{\text{smooth}} + I_{\text{singular}}.
```

### 4.1 Smooth Product-Quadrature Part

For the same triangle, it still performs product quadrature with `greens_smooth`:

```math
I_{\text{smooth}}
=\sum_{q_m,q_n}
\left[f_m\cdot f_n\,G_{\text{smooth}} - \frac{(\nabla\cdot f_m)(\nabla\cdot f_n)}{k^2}G_{\text{smooth}}\right]
 w_{q_m}w_{q_n}(2A)^2.
```

### 4.2 Singular Part

For each outer quadrature point `r_m`:

1. compute `S = analytical_integral_1overR(r_m, V1, V2, V3)`,
2. set `inner_scalar = S/(4π)`,
3. scalar singular term:

```math
\text{scl}_{\text{sing}} = \frac{(\nabla\cdot f_m)(\nabla\cdot f_n)}{k^2}\,\frac{S}{4\pi},
```

4. vector singular leading term:

```math
\text{vec}_{\text{lead}} = (f_m(r_m)\cdot f_n(r_m))\,\frac{S}{4\pi},
```

5. vector remainder (bounded integrand):

```math
\text{vec}_{\text{rem}}
= \int_T f_m(r_m)\cdot\frac{f_n(r')-f_n(r_m)}{4\pi|r_m-r'|}\, dS'.
```

The remainder is computed by quadrature and skips the `R=0` sample explicitly.

---

## 5. Adjacent-Cell Contribution Algorithm

`adjacent_cell_contribution(...)` handles edge-sharing triangle pairs where `1/R` is near-singular (bounded but poorly approximated by low-order quadrature). It uses the same `G = G_smooth + 1/(4πR)` kernel split as `self_cell_contribution`.

### 5.1 Smooth Part

Standard `Nq × Nq` product quadrature with `greens_smooth` — identical in form to the self-cell smooth part, but evaluated over two distinct triangles `tm` and `tn`.

### 5.2 Singular `1/(4πR)` Part

For each outer quadrature point `r_m` on `tm`:

1. **Scalar potential** (exact): `S = analytical_integral_1overR(r_m, V1_n, V2_n, V3_n)` — the same analytical formula used for self-cells. The full Graglia (1993) formula with off-plane `arctan` correction is used, so this is exact for both coplanar and non-coplanar adjacent triangles.

2. **Vector potential** (high-order quadrature): the inner integral `∫ f_n(r') / (4π|r_m - r'|) dS'` is evaluated with order-7 quadrature on `tn`. Unlike the self-cell case, `r_m` lies on the adjacent triangle (not on `tn`), so `1/R > 0` everywhere and the integrand is bounded. High-order quadrature is needed because `1/R` is large near the shared edge.

### 5.3 Adjacent-Cell Detection

`EFIEApplyCache` builds an edge-to-triangle map at construction time and stores all adjacent pairs in a `Set{NTuple{2,Int}}`. The function `_is_adjacent(cache, t1, t2)` performs O(1) lookup.

---

## 6. How EFIE Assembly Uses It

In `_efie_entry` (`src/assembly/EFIE.jl`), three cases are distinguished:

- if `tm == tn`: call `self_cell_contribution(...)`,
- else if `_is_adjacent(cache, tm, tn)`: call `adjacent_cell_contribution(...)`,
- else: use standard product quadrature with `greens(rm, rn, k)`.

Final scaling is applied after accumulation:

```math
Z_{mn} = -i\,\omega\mu_0\,(\text{accumulated }(\text{vec}-\text{scl})).
```

with `omega_mu0 = k * eta0`.

---

## 7. Current Scope and Limitations

1. **Implemented**: exact self-cell extraction for `tm == tn`.
2. **Implemented**: adjacent-cell (edge-sharing) singularity subtraction with high-order inner quadrature.
3. **Implemented**: full off-plane `analytical_integral_1overR` (Graglia 1993 / Wilton et al. 1984) with `arctan` correction terms. Works for both coplanar and non-coplanar triangle pairs.
4. **Not special-cased**: vertex-touching (but non-edge-sharing) triangle pairs.
5. **Not implemented**: dedicated near-singular quadrature for close but non-touching interactions, or Duffy-transform path.

---

## 8. Practical Verification Checks

A minimal self-consistency workflow is:

```julia
using DiffMoM, LinearAlgebra

mesh = make_rect_plate(0.1, 0.1, 4, 4)
rwg = build_rwg(mesh)
k = 2π / 0.1

# Dense EFIE build should complete with finite values
Z = assemble_Z_efie(mesh, rwg, k; mesh_precheck=false)
@assert all(isfinite, real.(Z))
@assert all(isfinite, imag.(Z))

# Symmetry-style sanity (integration noise tolerated)
rel_sym = norm(Z - transpose(Z)) / max(norm(Z), 1e-30)
println("relative transpose-mismatch = ", rel_sym)
```

For package-level regression, use:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## 9. Code Mapping

| Concept | Source | Function |
|---|---|---|
| Full free-space kernel | `src/basis/Greens.jl` | `greens` |
| Smooth split kernel | `src/basis/Greens.jl` | `greens_smooth` |
| Analytical `∫ 1/R` over triangle | `src/assembly/SingularIntegrals.jl` | `analytical_integral_1overR` |
| Self-cell regularized integral | `src/assembly/SingularIntegrals.jl` | `self_cell_contribution` |
| Adjacent-cell near-singular integral | `src/assembly/SingularIntegrals.jl` | `adjacent_cell_contribution` |
| Self/adjacent/non-adjacent branch | `src/assembly/EFIE.jl` | `_efie_entry` |
| Adjacent-pair detection | `src/assembly/EFIE.jl` | `_build_efie_cache`, `_is_adjacent` |
| Top-level dense EFIE assembly | `src/assembly/EFIE.jl` | `assemble_Z_efie` |

---

## 10. Exercises

1. Numerically verify `greens_smooth(r, rp, k)` approaches `-im*k/(4π)` as `|r-rp| -> 0`.
2. For a fixed triangle and point `P`, compare `analytical_integral_1overR` against high-order numerical integration with `P` slightly off-plane.
3. Measure how EFIE matrix entries change when increasing `quad_order` from `3` to `7` on a small mesh.

---

## 11. Chapter Checklist

- [ ] Explain why `1/R` needs special handling for self and adjacent terms.
- [ ] Derive and interpret `G = G_smooth + 1/(4πR)`.
- [ ] Map `analytical_integral_1overR` inputs to triangle geometry.
- [ ] Describe `self_cell_contribution` as smooth + singular + remainder parts.
- [ ] Describe `adjacent_cell_contribution` and explain why high-order inner quadrature suffices (bounded integrand).
- [ ] Locate the three-way branch in `_efie_entry` (self / adjacent / standard).
- [ ] Identify remaining limitations (vertex-touching pairs, near-singular non-touching).

---

## 12. Further Reading

- Graglia, R. D. (1993). Numerical integration of shape functions times 3D Green's function on triangles.
- Khayat, M. A., & Wilton, D. R. (2005). Numerical evaluation of singular and near-singular potential integrals.
- Gibson, W. C. *The Method of Moments in Electromagnetics* (singular-integral treatment chapters).

---

*Next: [Conditioning and Preconditioning](05-conditioning-preconditioning.md) covers system conditioning diagnostics and solver preconditioning pathways.*
