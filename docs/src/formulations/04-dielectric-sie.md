# Dielectric Surface Integral Equations (PMCHWT / Müller)

## Purpose

A perfect electric conductor scatters with a single equivalent electric current and one boundary integral equation. A *penetrable* dielectric body does not: the field exists on both sides of the boundary, so the scattering problem is governed by *two* equivalent currents (electric and magnetic) coupled across the surface. This chapter covers the closed-surface dielectric surface integral equation (SIE) solver in `DiffMoM.jl`, which discretizes a homogeneous isotropic body with stacked RWG electric and magnetic currents `[J; M]` and solves the resulting $2N \times 2N$ block system.

Two equivalent formulations of the same boundary-value problem are implemented. **PMCHWT** (Poggio-Miller-Chang-Harrington-Wu-Tsai) is a *first-kind* system whose blocks are all weakly singular integral operators. **Müller** is a *second-kind* system that adds an identity (Gram) term to the diagonal and is consequently better conditioned. Because they discretize the same physics, the two solvers return the same currents `J`, `M` (validated to under 1% on a dielectric sphere) — they differ only in the algebraic structure of the matrix and therefore in conditioning. This chapter derives both, explains the role of the principal-value magnetic-field operator and its second-kind identity residue, and walks through a runnable solve of both formulations on a closed mesh.

---

## Learning Goals

After this chapter, you should be able to:

1. State the surface-equivalence model that turns a penetrable dielectric into two equivalent currents `[J; M]` on the boundary.
2. Write the $2N \times 2N$ block system $A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}$ and identify each block as a row-weighted EFIE $T$ operator or a magnetic-field $K$ operator.
3. Derive the medium wave constants $k = k_0\sqrt{\varepsilon_r \mu_r}$ and $\eta = \eta_0\sqrt{\mu_r/\varepsilon_r}$ and explain the $1/\eta$ scaling on the magnetic row.
4. Explain the principal-value magnetic-field operator $K$ and the second-kind $\hat{\mathbf{n}}\times$ Gram identity residue it omits.
5. Distinguish PMCHWT (unit row weights, Gram term cancels) from Müller ($\mu/\varepsilon$-weighted rows, active Gram term).
6. Use `solve_dielectric_sie_3d`, `assemble_pmchwt_3d`, and `assemble_muller_3d`, and choose the dense direct or matrix-free GMRES path.
7. Recognize the closed-surface, non-periodic mesh requirements and the singular cases that the code rejects.

---

## 1. From a Penetrable Body to Two Surface Currents

### 1.1 The Surface-Equivalence Principle

Consider a homogeneous isotropic dielectric body occupying an interior region (constants $\varepsilon_r^{\text{int}}, \mu_r^{\text{int}}$) embedded in an exterior background (typically vacuum, $\varepsilon_r^{\text{ext}} = \mu_r^{\text{ext}} = 1$). The incident plane wave excites fields on *both* sides of the closed boundary $\Gamma$.

The surface-equivalence theorem replaces the volumetric material contrast by a pair of equivalent currents living on $\Gamma$:

```math
\mathbf{J} = \hat{\mathbf{n}} \times \mathbf{H}, \qquad
\mathbf{M} = -\hat{\mathbf{n}} \times \mathbf{E},
```

an electric surface current $\mathbf{J}$ and a magnetic surface current $\mathbf{M}$. Each region's fields can then be reconstructed from $\mathbf{J}$ and $\mathbf{M}$ through that region's homogeneous Green function. Enforcing continuity of the tangential fields across $\Gamma$ — separately on the exterior side and the interior side — produces *two* coupled boundary integral equations in the *two* unknown currents. Both currents are expanded in the same RWG basis $\{\mathbf{f}_n\}_{n=1}^{N}$, so each carries $N$ = `rwg.nedges` coefficients.

### 1.2 The Stacked Block System

Stacking the electric-current coefficients on top of the magnetic-current coefficients, $\mathbf{x} = [\mathbf{J};\, \mathbf{M}]$, the discretized system is a $2N \times 2N$ block matrix:

```math
\begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}
\begin{bmatrix} \mathbf{J} \\ \mathbf{M} \end{bmatrix}
=
\begin{bmatrix} \mathbf{v}_E \\ \mathbf{v}_H \end{bmatrix}.
```

The two block rows are the *electric-field row* (E-row, top) and the *magnetic-field row* (H-row, bottom):

- $A_{11}, A_{22}$ — the **diagonal** $T$ blocks, built from the singularity-treated EFIE operator, acting on the like-kind current of each row.
- $A_{12}, A_{21}$ — the **off-diagonal** $K$ blocks, built from the magnetic-field operator, coupling each row to the opposite current.

The right-hand side $[\mathbf{v}_E;\, \mathbf{v}_H]$ tests the incident electric and magnetic fields, which live only in the exterior region. The rest of this chapter derives the entries of these four blocks.

---

## 2. Medium Constants and Time Convention

`DiffMoM.jl` uses the $e^{+i\omega t}$ time convention. With this choice the free-space scalar Green function and the incident plane wave are

```math
G(\mathbf{r}, \mathbf{r}') = \frac{e^{-ikR}}{4\pi R}, \quad R = |\mathbf{r} - \mathbf{r}'|, \qquad
\mathbf{E}^{\text{inc}}(\mathbf{r}) = \hat{\mathbf{p}}\, E_0\, e^{-i\mathbf{k}\cdot\mathbf{r}}.
```

(The convention is declared in `src/basis/Greens.jl` and the plane wave in `src/assembly/Excitation.jl`.) Each homogeneous region carries its own wave constants. For a region with relative parameters $\varepsilon_r, \mu_r$, the medium wavenumber and intrinsic impedance are

```math
k = k_0 \sqrt{\varepsilon_r \mu_r}, \qquad
\eta = \eta_0 \sqrt{\frac{\mu_r}{\varepsilon_r}},
```

where $k_0$ is the free-space wavenumber and $\eta_0 = 376.730313668\;\Omega$ is the free-space impedance. The helper `dielectric_medium_3d(k0, eps_r, mu_r)` packages these into a `DielectricMedium3D` struct with fields `eps_r`, `mu_r`, `k`, `eta`. It validates the inputs: `k0 > 0`, `eta0 > 0`, and both $\varepsilon_r$, $\mu_r$ finite and nonzero (a zero or infinite parameter makes $k$ or $\eta$ ill-defined and raises an error).

For a lossy interior the parameters are complex; the square roots above then carry the medium loss into both $k$ (attenuation) and $\eta$ (a complex wave impedance), as the worked example below shows.

---

## 3. The Diagonal $T$ Blocks (Row-Weighted EFIE)

### 3.1 Reusing the EFIE Operator

The diagonal blocks are the EFIE $T$ operator — the same singularity-treated impedance operator used for PEC scattering (see [Assembly and Solve](../api/assembly-solve.md)). For a region with wavenumber $k$ and impedance $\eta$ it is, symbolically,

```math
\big(T \mathbf{a}\big)_m = \Big\langle \mathbf{f}_m,\; i k \eta \!\int_\Gamma \! \Big(\mathbf{f}_n + \tfrac{1}{k^2}\nabla(\nabla'\!\cdot\mathbf{f}_n)\Big) G\, dS' \Big\rangle,
```

assembled by `assemble_Z_efie` with built-in self/near singularity treatment. Each region contributes its own $T$ block (one for the exterior wavenumber/impedance, one for the interior), and the diagonal block of each row is a weighted sum of the two:

```math
A_{11} = c^{E}_{\text{ext}}\, T^{\text{ext}}_{E} + c^{E}_{\text{int}}\, T^{\text{int}}_{E}, \qquad
A_{22} = c^{H}_{\text{ext}}\, T^{\text{ext}}_{H} + c^{H}_{\text{int}}\, T^{\text{int}}_{H}.
```

The row weights $c$ are what distinguish PMCHWT from Müller (Section 6).

### 3.2 Why the Magnetic Row Uses $1/\eta$

The E-row diagonal is the ordinary EFIE $T$ operator built with the medium impedance $\eta$ (passed as `eta0=exterior.eta` / `eta0=interior.eta` in the assembly call). The H-row tests the magnetic field and acts on the *magnetic* current, which by electric-magnetic duality is governed by the *dual* operator. In the implementation this dual operator is produced by reusing the same EFIE assembly with the **inverse** impedance (`eta0 = 1/exterior.eta`, `1/interior.eta`):

```math
T^{\text{ext}}_{E} \;\propto\; \eta_{\text{ext}}, \qquad
T^{\text{ext}}_{H} \;\propto\; \frac{1}{\eta_{\text{ext}}}.
```

This $1/\eta$ scaling on the magnetic row is exactly what converts the electric $T$ operator into the correct magnetic-current $T$ operator. The same singularity-treated kernel is reused, so the magnetic row inherits the EFIE's accurate self-cell integration for free.

---

## 4. The Off-Diagonal $K$ Blocks (Magnetic-Field Operator)

### 4.1 The Principal-Value $K$ Operator

The off-diagonal blocks couple each row to the opposite current through the **magnetic-field operator** $K$, the tested principal-value integral of the magnetic field radiated by a surface current:

```math
K[m,n] = \Big\langle \mathbf{f}_m,\; \text{PV}\!\int_\Gamma \nabla G(\mathbf{r},\mathbf{r}') \times \mathbf{f}_n(\mathbf{r}')\, dS' \Big\rangle .
```

It is assembled by `assemble_magnetic_field_operator_3d` using *product* (double) quadrature over panel pairs. The per-pair kernel evaluated at quadrature points $(\mathbf{r}_m, \mathbf{r}_n)$ is

```math
\mathbf{f}_m \cdot \big(\nabla G(\mathbf{r}_m,\mathbf{r}_n) \times \mathbf{f}_n\big),
\qquad
\nabla G = \Big(-ik - \tfrac{1}{R}\Big) G\, \hat{\mathbf{R}}, \;\; \hat{\mathbf{R}} = \frac{\mathbf{r}-\mathbf{r}'}{R},
```

with the RWG Jacobian factor $2A$ applied on *each* panel of the pair. As with the $T$ blocks, each region contributes a $K$ block at its own wavenumber.

### 4.2 Sign Convention on the Rows

The two off-diagonal blocks carry the weighted $K$ operator with opposite signs — **minus on the E-row, plus on the H-row**:

```math
A_{12} = -\big(c^{E}_{\text{ext}}K^{\text{ext}} + c^{E}_{\text{int}}K^{\text{int}}\big) + (\text{Gram}), \qquad
A_{21} = +\big(c^{H}_{\text{ext}}K^{\text{ext}} + c^{H}_{\text{int}}K^{\text{int}}\big) + (\text{Gram}),
```

where the Gram contribution is explained next.

### 4.3 Near-Singular Quadrature Promotion

The $1/R$ factor in $\nabla G$ makes the $K$ kernel near-singular when the two panels are close. To control this, the assembly promotes any triangle pair sharing **at least one vertex** — self, edge-adjacent, *and* vertex-touching — to the higher-order `singular_quad_order` rule (default 7); all other (far) pairs use `quad_order` (default 3). Promoting the vertex-touching pairs, not just edge-sharing ones, measurably reduces the $K$ quadrature error (the source comment records roughly $0.99\% \to 0.27\%$ on an icosphere reference). The available rules are restricted to the package's quadrature orders (1, 3, 4, 7).

---

## 5. The Second-Kind Identity Residue ($\hat{\mathbf{n}}\times$ Gram)

### 5.1 The Principal Value Omits a Residue

When the field point $\mathbf{r}$ approaches the surface, the magnetic-field integral splits into a principal-value part plus a *residue* arising from the singular self-term. The operator $K$ above keeps only the principal value, so the discretization is missing this residue. The missing piece is a pure identity-like overlap term — the **$\hat{\mathbf{n}}\times$ Gram matrix**:

```math
G_{\hat{\mathbf{n}}}[m,n] = \big\langle \mathbf{f}_m,\; \hat{\mathbf{n}} \times \mathbf{f}_n \big\rangle
= \int_{\Gamma} \mathbf{f}_m \cdot (\hat{\mathbf{n}} \times \mathbf{f}_n)\, dS,
```

integrated only over the triangle(s) the two RWG functions *both* support, with the outward triangle normal $\hat{\mathbf{n}}$ (again with the $2A$ Jacobian). It is assembled by `_ncross_gram_3d`. This is a sparse, identity-like overlap matrix — the hallmark of a second-kind operator.

### 5.2 Where It Enters

The residue appears on the off-diagonal blocks with coefficient $(c_{\text{ext}} - c_{\text{int}})\cdot\tfrac{1}{2}$ for each row:

```math
A_{12} = -\Big(K_E + (c^{E}_{\text{ext}} - c^{E}_{\text{int}})\,\tfrac{1}{2}\, G_{\hat{\mathbf{n}}}\Big),
\qquad
A_{21} = +\Big(K_H + (c^{H}_{\text{ext}} - c^{H}_{\text{int}})\,\tfrac{1}{2}\, G_{\hat{\mathbf{n}}}\Big).
```

The coefficient is a *difference* of the exterior and interior row weights. This is the crucial structural fact: **whether the residue survives depends entirely on the row weights**, which is precisely the difference between PMCHWT and Müller. When the two weights are equal the coefficient vanishes and the Gram matrix is skipped entirely.

---

## 6. PMCHWT vs Müller: The Row Weights

The four row weights $c^{E}_{\text{ext}}, c^{E}_{\text{int}}, c^{H}_{\text{ext}}, c^{H}_{\text{int}}$ select the formulation. They multiply *both* the diagonal $T$ block and the off-diagonal $K$ block of each region, and they control the Gram coefficient through their differences.

### 6.1 PMCHWT — First Kind

PMCHWT uses **unit row weights**:

```math
c^{E}_{\text{ext}} = c^{E}_{\text{int}} = c^{H}_{\text{ext}} = c^{H}_{\text{int}} = 1.
```

The diagonal $T$ blocks add the exterior and interior EFIE blocks with equal weight, and — because the exterior and interior coefficients are *equal* in each row — the Gram coefficient $(c_{\text{ext}} - c_{\text{int}})\cdot\tfrac{1}{2}$ is **identically zero**. The implementation takes the equal-coefficient branch and skips the Gram matrix entirely. The off-diagonal blocks are then just $A_{12} = -K_E$, $A_{21} = +K_H$. Every block of PMCHWT is a weakly singular integral operator with no identity term — this is what makes PMCHWT a *first-kind* system.

### 6.2 Müller — Second Kind

Müller uses **$\mu/\varepsilon$-weighted row coefficients**:

```math
c^{E}_{\text{ext}} = \frac{\mu^{\text{int}}_r}{\mu^{\text{ext}}_r + \mu^{\text{int}}_r}, \quad
c^{E}_{\text{int}} = \frac{\mu^{\text{ext}}_r}{\mu^{\text{ext}}_r + \mu^{\text{int}}_r},
```
```math
c^{H}_{\text{ext}} = \frac{\varepsilon^{\text{int}}_r}{\varepsilon^{\text{ext}}_r + \varepsilon^{\text{int}}_r}, \quad
c^{H}_{\text{int}} = \frac{\varepsilon^{\text{ext}}_r}{\varepsilon^{\text{ext}}_r + \varepsilon^{\text{int}}_r}.
```

Now the exterior and interior weights *differ* in each row, so the Gram coefficient is nonzero and the $\hat{\mathbf{n}}\times$ Gram identity term is added on the off-diagonal blocks. This identity term dominates the diagonal of the system, converting it into a *second-kind* (identity + compact) operator. The right-hand side must be scaled consistently by the exterior row weights, $[\,c^{E}_{\text{ext}}\,\mathbf{v}_E;\; c^{H}_{\text{ext}}\,\mathbf{v}_H\,]$, which is why the Müller RHS assembly requires the interior medium (to compute the weights). The denominators $\mu^{\text{ext}}_r + \mu^{\text{int}}_r$ and $\varepsilon^{\text{ext}}_r + \varepsilon^{\text{int}}_r$ must be nonzero — the code raises an error if either vanishes (the Müller weights are singular there).

### 6.3 Same Physics, Different Conditioning

PMCHWT and Müller are **distinct matrices** (the tests assert their norm difference exceeds $10^{-4}$; the worked example below sees roughly 48% relative difference). But they discretize the *same* boundary-value problem, so the solved currents `J`, `M` coincide. The trade-off is conditioning: PMCHWT is first-kind (no compact-resolvent structure), while Müller is second-kind and better conditioned because the identity Gram term dominates the diagonal. **The $\hat{\mathbf{n}}\times$ Gram term is not optional cosmetics** — without it the Müller currents disagree with PMCHWT by 20–50% (over 100% for the magnetic current), as the validation testset records.

---

## 7. The Right-Hand Side

The incident plane wave exists only in the exterior region, so the forcing tests the incident exterior fields. The E-row tests the incident electric field directly via `assemble_excitation`, giving $\mathbf{v}_E$. The H-row tests the incident magnetic field

```math
\mathbf{H}^{\text{inc}} = \frac{\hat{\mathbf{k}} \times \mathbf{E}^{\text{inc}}}{\eta_{\text{ext}}},
```

with a leading minus sign in the tested forcing, $\mathbf{v}_H[n] \mathrel{+}= -w_q\,\langle \mathbf{f}_n, \mathbf{H}^{\text{inc}}\rangle\, 2A$. For PMCHWT the RHS is $[\mathbf{v}_E;\, \mathbf{v}_H]$; for Müller it is scaled by the exterior row weights to stay consistent with the weighted block matrix (Section 6.2). The plane-wave solve overload supplies the interior medium automatically so the Müller scaling can be computed.

---

## 8. Matrix-Free Operator

For iterative (GMRES) solves the system can be applied without forming any dense $N \times N$ block. `matrixfree_dielectric_sie_operator_3d` returns a `MatrixFreeDielectricSIE3D` that wraps matrix-free EFIE operators (`Ze`/`Zh` for exterior/interior) and matrix-free magnetic-field operators (`K`), applies the row weights, and supports the 5-argument `mul!`. The off-diagonal Gram coefficients are stored as

```math
c_{g,E} = -\,(c^{E}_{\text{ext}} - c^{E}_{\text{int}})\cdot\tfrac{1}{2}, \qquad
c_{g,H} = +\,(c^{H}_{\text{ext}} - c^{H}_{\text{int}})\cdot\tfrac{1}{2},
```

matching the dense block signs of Section 5.2. The dense $\hat{\mathbf{n}}\times$ Gram matrix is precomputed *only when nonzero* — i.e. for Müller; PMCHWT stores a $0\times0$ placeholder and the matvec skips the Gram contribution. The dense and matrix-free operators are validated to agree to machine precision ($< 10^{-13}$).

---

## 9. Worked Example

The script below solves a lossy dielectric body in vacuum with both PMCHWT and Müller, on a tiny closed tetrahedron (4 triangles, $N=6$, system size $2N=12$) so it runs in seconds. It demonstrates the full workflow: building a closed-surface RWG basis, deriving the medium constants, solving both formulations, confirming the matrices are distinct yet yield the same currents, and matching the matrix-free GMRES path to the dense direct solve.

No exported closed-mesh generator exists, so the helper `oriented_tetrahedron_mesh` is inlined verbatim; it orients every face outward so the surface is genuinely closed. For a quantitative accuracy study you would use a refined closed surface such as an icosphere — see Section 10.

Run it with `julia --project=/path/to/DiffMoM.jl thisfile.jl`.

```julia
using DiffMoM
using LinearAlgebra

# --- Closed-surface mesh helper (inlined; no exported generator exists) ---
function oriented_tetrahedron_mesh()
    verts = Vec3[
        Vec3(1.0, 1.0, 1.0), Vec3(-1.0, -1.0, 1.0),
        Vec3(-1.0, 1.0, -1.0), Vec3(1.0, -1.0, -1.0),
    ]
    faces = [(1, 2, 3), (1, 4, 2), (1, 3, 4), (2, 4, 3)]
    tri = zeros(Int, 3, length(faces))
    for (t, f) in enumerate(faces)
        inds = collect(f)
        a, bb, c = verts[inds[1]], verts[inds[2]], verts[inds[3]]
        n = cross(bb - a, c - a); center = (a + bb + c) / 3
        dot(n, center) < 0 && ((inds[2], inds[3]) = (inds[3], inds[2]))
        tri[:, t] .= inds
    end
    return TriMesh(hcat(verts...), tri)
end

# --- Build the closed mesh and RWG basis (must be closed, non-periodic) ---
mesh = oriented_tetrahedron_mesh()
rwg  = build_rwg(mesh; allow_boundary=false, require_closed=true)
N    = rwg.nedges
println("mesh: ", ntriangles(mesh), " triangles, RWG edges N = ", N,
        "  (system size 2N = ", 2N, ")")

# --- Problem: dielectric body in vacuum, plane wave incidence ---
k0     = 0.7                 # free-space wavenumber (rad/m)
eps_in = 2.2 - 0.03im        # interior relative permittivity (lossy)
mu_in  = 1.3 - 0.02im        # interior relative permeability (lossy)
pw     = make_plane_wave(Vec3(0.0, 0.0, k0), 1.0, Vec3(1.0, 0.0, 0.0))

# Derived medium constants: k = k0*sqrt(eps_r*mu_r), eta = eta0*sqrt(mu_r/eps_r)
ext = dielectric_medium_3d(k0)
int = dielectric_medium_3d(k0, eps_in, mu_in)
println("exterior k = ", round(ext.k, digits=4), ", eta = ", round(ext.eta, digits=2))
println("interior k = ", round(int.k, digits=4), ", eta = ", round(int.eta, digits=2))

# --- Solve BOTH formulations from the same plane wave ---
res_pm = solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, pw;
                                 mur_in=mu_in, formulation=:pmchwt,
                                 quad_order=3, singular_quad_order=7)
res_mu = solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, pw;
                                 mur_in=mu_in, formulation=:muller,
                                 quad_order=3, singular_quad_order=7)

println("PMCHWT: formulation=", res_pm.formulation,
        ", |J|=", round(norm(res_pm.J), digits=5),
        ", |M|=", round(norm(res_pm.M), digits=5))
println("Muller: formulation=", res_mu.formulation,
        ", |J|=", round(norm(res_mu.J), digits=5),
        ", |M|=", round(norm(res_mu.M), digits=5))

# --- The two systems are DIFFERENT matrices but solve the SAME BVP ---
A_pm = assemble_pmchwt_3d(mesh, rwg, k0, eps_in; mur_in=mu_in,
                          quad_order=3, singular_quad_order=7)
A_mu = assemble_muller_3d(mesh, rwg, k0, eps_in; mur_in=mu_in,
                          quad_order=3, singular_quad_order=7)
println("matrix difference ||A_pm - A_mu|| / ||A_pm|| = ",
        round(norm(A_pm - A_mu) / norm(A_pm), digits=4), " (distinct formulations)")

# Each solved current satisfies its own (weighted) system to solver tolerance.
resid_pm = norm(res_pm.A * vcat(res_pm.J, res_pm.M) - res_pm.rhs) / max(norm(res_pm.rhs), eps())
resid_mu = norm(res_mu.A * vcat(res_mu.J, res_mu.M) - res_mu.rhs) / max(norm(res_mu.rhs), eps())
println("PMCHWT relative residual = ", resid_pm)
println("Muller relative residual = ", resid_mu)

# --- Currents agree: PMCHWT vs Muller discretize the same BVP ---
relJ = norm(res_mu.J - res_pm.J) / norm(res_pm.J)
relM = norm(res_mu.M - res_pm.M) / norm(res_pm.M)
println("PMCHWT vs Muller current agreement: relJ = ", round(relJ, digits=4),
        ", relM = ", round(relM, digits=4))
println("(On a refined dielectric SPHERE the test asserts relJ, relM < 1%, ",
        "tightening under refinement; the tiny tetra is coarser.)")

# --- Matrix-free GMRES path matches the dense direct solve ---
res_gmres = solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, pw;
                                    mur_in=mu_in, formulation=:pmchwt,
                                    solver=:gmres, quad_order=3,
                                    singular_quad_order=7, tol=1e-12, maxiter=200)
gmres_match = norm(vcat(res_gmres.J, res_gmres.M) - vcat(res_pm.J, res_pm.M)) /
              norm(vcat(res_pm.J, res_pm.M))
println("GMRES vs direct (PMCHWT) relative difference = ", gmres_match,
        "  (A_LU===nothing: ", res_gmres.A_LU === nothing, ")")

println("DONE")
```

### 9.1 What the Output Shows

Running the script prints (values from an actual run):

```
mesh: 4 triangles, RWG edges N = 6  (system size 2N = 12)
exterior k = 0.7 + 0.0im, eta = 376.73 + 0.0im
interior k = 1.1838 - 0.0172im, eta = 289.6 - 0.25im
PMCHWT: formulation=pmchwt, |J|=0.00364, |M|=1.09005
Muller: formulation=muller, |J|=0.00362, |M|=1.16338
matrix difference ||A_pm - A_mu|| / ||A_pm|| = 0.4833 (distinct formulations)
PMCHWT relative residual = 5.79e-16
Muller relative residual = 3.43e-16
PMCHWT vs Muller current agreement: relJ = 0.0433, relM = 0.0858
GMRES vs direct (PMCHWT) relative difference = 2.70e-15  (A_LU===nothing: true)
DONE
```

Key observations:

- The lossy interior produces a **complex** wavenumber ($k = 1.18 - 0.017i$, the negative imaginary part being attenuation under $e^{+i\omega t}$) and a **complex** impedance ($\eta = 289.6 - 0.25i$).
- PMCHWT and Müller are genuinely different matrices ($\approx 48\%$ relative difference) yet each solution satisfies its own system to machine precision ($\sim 10^{-16}$).
- The two formulations' currents agree to a few percent on this *very coarse* 4-triangle mesh ($\text{relJ} \approx 4\%$, $\text{relM} \approx 9\%$). On a refined sphere the validation testset asserts agreement under 1% (Section 10) — the coarse agreement here reflects discretization error, not a formulation error.
- The matrix-free GMRES path reproduces the dense direct solve to $\sim 10^{-15}$, and returns `A_LU === nothing` (no dense factorization is formed for the iterative path).

---

## 10. Validation

This subsystem has no standalone `validation/` script; its validation lives entirely in the test suite at `test/test_surface_ie3d.jl` (registered in `test/runtests.jl`). Two testsets are decisive:

**`Dielectric 3D SIE assembly/solve`** checks the algebraic machinery:

- The dense and matrix-free $K$ operators agree to $< 10^{-13}$ (`mul!` vs dense).
- The dense and matrix-free PMCHWT operators agree to $< 10^{-13}$.
- PMCHWT and Müller are **distinct** matrices (norm difference $> 10^{-4}$).
- A zero RHS yields zero currents with an exact residual.
- Direct and GMRES PMCHWT solves agree to $< 10^{-9}$.
- Plane-wave PMCHWT and Müller solves have machine-precision residuals.
- An **open** mesh (a rectangular plate) makes `assemble_pmchwt_3d` throw, and `formulation=:cfie` throws.

**`PMCHWT vs Muller currents agree (dielectric sphere)`** is the physics oracle. On a 1-subdivision icosphere, for two $(\varepsilon_{\text{int}}, \mu_{\text{int}})$ cases, it asserts the relative current mismatch $\text{relJ} < 1\%$ and $\text{relM} < 1\%$. The testset comment records that **without** the $\hat{\mathbf{n}}\times$ Gram identity term the mismatch is roughly 20–50% (over 100% for the magnetic current), confirming that the residue of Section 5 is essential for the two formulations to coincide. It also checks the Müller RHS-consistent residual ($< 10^{-10}$) and that the dense and matrix-free Müller operators are identical to $< 10^{-13}$.

The worked example in Section 9 is the coarse-mesh counterpart: it runs in seconds on a 4-triangle tetrahedron and reproduces every qualitative property (distinct matrices, machine-precision residuals, GMRES matching direct), with the percent-level current agreement tightening toward the test's sub-1% bound under mesh refinement.

---

## 11. When to Use / Limitations

**Use the dielectric SIE solver when** you scatter from a homogeneous, isotropic, *penetrable* body (a dielectric or magnetic object) bounded by a closed surface. Use **PMCHWT** as the well-understood reference formulation, and **Müller** when you want better-conditioned iterative convergence — both return the same currents.

**Hard requirements (the code enforces these):**

- **Closed surface, mandatory.** The mesh quality precheck runs with `allow_boundary=false, require_closed=true`; an open mesh (e.g. a plate) raises an error. Build the RWG basis the same way: `build_rwg(mesh; allow_boundary=false, require_closed=true)`.
- **Non-periodic, same-mesh RWG.** A periodic/Bloch basis (`rwg.has_periodic_bloch`) is rejected, and the RWG data must be built from the *same* `mesh` object passed to the solver (`mesh === rwg.mesh`).
- **Valid formulation symbol.** Only `:pmchwt` and `:muller` are accepted; anything else (e.g. `:cfie`) raises "Unsupported dielectric SIE formulation".
- **Müller-specific.** The Müller weights are singular when $\mu_{\text{ext}} + \mu_{\text{int}} = 0$ or $\varepsilon_{\text{ext}} + \varepsilon_{\text{int}} = 0$, and the code errors in those cases. Müller RHS assembly also requires the interior medium (the plane-wave overload supplies it automatically).
- **Finite, nonzero media.** `dielectric_medium_3d` requires `k0 > 0`, `eta0 > 0`, and both $\varepsilon_r, \mu_r$ finite and nonzero.

**Accuracy notes.** The sub-1% PMCHWT-vs-Müller agreement is a property of a *sufficiently refined* mesh; on a tiny tetrahedron the agreement is coarser (a few percent) purely because the discretization is coarse. Quadrature orders are restricted to the package's available rules (1, 3, 4, 7); the defaults `quad_order=3`, `singular_quad_order=7` are a good starting point.

---

## 12. Code Mapping

| Concept | Exported symbol | Source file |
|---------|-----------------|-------------|
| Medium constants $k, \eta$ | `dielectric_medium_3d`, `DielectricMedium3D` | `src/mom3d/SurfaceIE3D.jl` |
| Magnetic-field $K$ operator (dense) | `assemble_magnetic_field_operator_3d` | `src/mom3d/SurfaceIE3D.jl` |
| Magnetic-field $K$ operator (matrix-free) | `matrixfree_magnetic_field_operator_3d`, `MatrixFreeMagneticFieldOperator3D` | `src/mom3d/SurfaceIE3D.jl` |
| Right-hand side $[\mathbf{v}_E;\,\mathbf{v}_H]$ | `assemble_dielectric_sie_rhs_3d` | `src/mom3d/SurfaceIE3D.jl` |
| Generic block-system assembly | `assemble_dielectric_sie_3d` | `src/mom3d/SurfaceIE3D.jl` |
| PMCHWT system (first kind) | `assemble_pmchwt_3d` | `src/mom3d/SurfaceIE3D.jl` |
| Müller system (second kind) | `assemble_muller_3d` | `src/mom3d/SurfaceIE3D.jl` |
| Matrix-free $2N\times2N$ operator | `matrixfree_dielectric_sie_operator_3d`, `MatrixFreeDielectricSIE3D` | `src/mom3d/SurfaceIE3D.jl` |
| Solve (direct or GMRES) | `solve_dielectric_sie_3d`, `DielectricSIEResult3D` | `src/mom3d/SurfaceIE3D.jl` |
| $\hat{\mathbf{n}}\times$ Gram residue (internal) | `_ncross_gram_3d` | `src/mom3d/SurfaceIE3D.jl` |
| Row weights (internal) | `_surface_sie_coefficients_3d` | `src/mom3d/SurfaceIE3D.jl` |
| EFIE $T$ blocks (reused) | `assemble_Z_efie` | `src/assembly/` (see [Assembly and Solve](../api/assembly-solve.md)) |
| Scalar Green function / gradient | `greens`, `grad_greens` | `src/basis/Greens.jl` |
| Plane wave / excitation | `make_plane_wave`, `assemble_excitation` | `src/assembly/Excitation.jl` |

See the full API reference at [API: Dielectric Surface Integral Equation (3D)](../api/dielectric-sie-3d.md).

---

## 13. Exercises

### 13.1 Conceptual

1. **Two currents, one body.** Explain physically why a penetrable dielectric needs *both* an electric and a magnetic equivalent current, whereas a PEC needs only an electric current. Which boundary condition forces $\mathbf{M} = \mathbf{0}$ on a PEC?

2. **The $1/\eta$ row.** The H-row diagonal reuses the EFIE assembly with `eta0 = 1/eta` instead of `eta`. Argue from electric-magnetic duality why the inverse impedance produces the correct magnetic-current $T$ operator.

3. **Why the Gram term cancels for PMCHWT.** The off-diagonal Gram coefficient is $(c_{\text{ext}} - c_{\text{int}})/2$. Show that PMCHWT's unit weights make this zero, and that Müller's $\mu/\varepsilon$ weights make it nonzero. What would happen to the Müller solution if the term were omitted?

4. **First vs second kind.** Explain why adding the identity-like $\hat{\mathbf{n}}\times$ Gram term to the diagonal improves conditioning. Why is a first-kind system (PMCHWT) typically worse-conditioned than a second-kind one (Müller)?

### 13.2 Numerical Experiments

5. **Refinement study.** Replace the tetrahedron with a refined closed surface (an icosphere; the test suite uses `_icosphere_mesh` internally). Solve both formulations and confirm that `relJ` and `relM` shrink below 1% as you refine, matching the validation testset.

6. **Lossless vs lossy.** Re-run the worked example with a *lossless* interior (`eps_in = 2.2`, `mu_in = 1.3`, no imaginary part). Confirm that `int.k` and `int.eta` become real, and compare the currents to the lossy case.

7. **Quadrature promotion.** Assemble the $K$ operator with `singular_quad_order = 3` (no promotion) and with `singular_quad_order = 7`, and compare the resulting matrices. On a refined mesh, quantify the change in the PMCHWT-vs-Müller current agreement.

8. **GMRES conditioning.** Solve a moderately refined sphere with `solver=:gmres` for both PMCHWT and Müller at the same tolerance, and compare the GMRES iteration counts from `res.stats`. Does Müller converge in fewer iterations, consistent with its second-kind conditioning?

### 13.3 Error Handling

9. **Open mesh.** Build a rectangular plate, call `assemble_pmchwt_3d` on it, and confirm it throws. What single property of the mesh causes the failure?

10. **Singular Müller.** Construct media with $\varepsilon_{\text{ext}} + \varepsilon_{\text{int}} = 0$ (e.g. an idealized $\varepsilon_{\text{int}} = -1$ in vacuum) and confirm `assemble_muller_3d` raises the singular-weights error. Why is PMCHWT immune to this particular failure?

---

## 14. Further Reading

1. **Surface-equivalence and dielectric SIE formulations:**
   - Harrington, R. F. (1989). *Boundary integral formulations for homogeneous material bodies.* Journal of Electromagnetic Waves and Applications. The classic statement of PMCHWT.
   - Poggio, A. J., & Miller, E. K. (1973). *Integral equation solutions of three-dimensional scattering problems.* In Mittra, R. (Ed.), *Computer Techniques for Electromagnetics.*

2. **Müller formulation and conditioning:**
   - Müller, C. (1969). *Foundations of the Mathematical Theory of Electromagnetic Waves.* Springer. The original second-kind formulation.
   - Ylä-Oijala, P., & Taskinen, M. (2005). *Well-conditioned Müller formulation for electromagnetic scattering by dielectric objects.* IEEE Transactions on Antennas and Propagation.

3. **RWG basis and MoM:**
   - Rao, S. M., Wilton, D. R., & Glisson, A. W. (1982). *Electromagnetic scattering by surfaces of arbitrary shape.* IEEE Transactions on Antennas and Propagation.
   - Gibson, W. C. (2008). *The Method of Moments in Electromagnetics.* Chapman & Hall/CRC.

---

*See also: [API: Dielectric Surface Integral Equation (3D)](../api/dielectric-sie-3d.md) for full signatures, [Physical Optics Approximation](../methods/03-physical-optics.md) for the high-frequency PEC counterpart, and [Assembly and Solve](../api/assembly-solve.md) for the reused EFIE `T` blocks.*
