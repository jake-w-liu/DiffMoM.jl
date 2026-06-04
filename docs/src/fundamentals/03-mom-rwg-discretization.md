# MoM and RWG Discretization

## Purpose

This chapter explains how the continuous Electric Field Integral Equation (EFIE) is transformed into a finite linear system using the Method of Moments (MoM) with Rao‑Wilton‑Glisson (RWG) basis functions. We detail the mathematical foundations of Galerkin testing, derive the explicit form of RWG functions on triangular meshes, and connect each step to the implementation in `DiffMoM.jl`.

The RWG basis is the workhorse of surface integral equation solvers for electromagnetics. Its clever design ensures current continuity across element boundaries, enforces charge conservation, and yields simple analytical properties that enable efficient numerical integration. Understanding RWG functions is essential for interpreting MoM results, debugging assembly routines, and extending the codebase.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive the Galerkin discretization of the EFIE, starting from the weighted‑residual formulation and arriving at the linear system $\mathbf{Z}\mathbf{I} = \mathbf{V}$.
2. Write the mathematical definition of an RWG basis function on a pair of triangles and explain the geometric meaning of each term.
3. Prove that RWG functions are divergence‑conforming (current continuity) and compute their surface divergence explicitly.
4. Map the mixed‑potential EFIE matrix elements $Z_{mn}$ to the assembly loops and quadrature routines in the code.
5. Explain why RWG functions lead to a simple, constant divergence on each triangle and how this property simplifies the scalar‑potential term.
6. Use the package's inspection tools to examine RWG data structures and verify geometric consistency.

---

## 1. Introduction to the Method of Moments (MoM)

The Method of Moments is a general technique for converting linear operator equations into finite linear systems. For the EFIE, we start with the operator equation (including impedance boundary condition)

```math
\mathcal{T}[\mathbf{J}](\mathbf{r}) - Z_s(\mathbf{r})\mathbf{J}(\mathbf{r}) = -\mathbf{E}_t^{\mathrm{inc}}(\mathbf{r}), \qquad \mathbf{r} \in \Gamma,
```

where $\mathcal{T}$ is the EFIE integral operator defined in Chapter 2. The unknown surface current $\mathbf{J}$ is approximated by a finite expansion in basis functions $\{\mathbf{f}_n\}_{n=1}^N$:

```math
\mathbf{J}(\mathbf{r}) \approx \sum_{n=1}^N I_n \mathbf{f}_n(\mathbf{r}).
```

The coefficients $I_n$ are complex numbers representing the strength of current associated with each basis function.

### 1.1 Galerkin Testing (Weighted Residual Method)

To determine the $N$ unknown coefficients, we enforce the residual—the difference between the left‑hand and right‑hand sides of the operator equation—to be orthogonal to a set of **testing functions**. In the Galerkin approach, we choose the testing functions to be the same as the basis functions. This yields the system of equations

```math
\langle \mathbf{f}_m, \mathcal{T}[\mathbf{J}] - Z_s\mathbf{J} \rangle = -\langle \mathbf{f}_m, \mathbf{E}_t^{\mathrm{inc}} \rangle, \qquad m = 1,\dots,N.
```

Substituting the expansion for $\mathbf{J}$ and using linearity gives

```math
\sum_{n=1}^N \bigl( \langle \mathbf{f}_m, \mathcal{T}[\mathbf{f}_n] \rangle - \langle \mathbf{f}_m, Z_s\mathbf{f}_n \rangle \bigr) I_n = -\langle \mathbf{f}_m, \mathbf{E}_t^{\mathrm{inc}} \rangle.
```

Defining the **impedance matrix** $\mathbf{Z}$ and the **excitation vector** $\mathbf{V}$ by

```math
Z_{mn} = \langle \mathbf{f}_m, \mathcal{T}[\mathbf{f}_n] \rangle - \langle \mathbf{f}_m, Z_s\mathbf{f}_n \rangle, \qquad
V_m = -\langle \mathbf{f}_m, \mathbf{E}_t^{\mathrm{inc}} \rangle,
```

we obtain the linear system

```math
\mathbf{Z} \mathbf{I} = \mathbf{V}.
```

This is the discrete problem that `DiffMoM.jl` assembles and solves.

### 1.2 Choice of Basis Functions

The accuracy and efficiency of MoM depend critically on the choice of basis functions $\{\mathbf{f}_n\}$. Ideal basis functions for surface current should:

1. **Approximate smooth currents well** (low‑order polynomials often suffice).
2. **Enforce current continuity** across element boundaries (no artificial charge accumulation).
3. **Have local support** to produce sparse coupling patterns (though the EFIE operator itself is dense).
4. **Lead to simple integrals** for efficient numerical evaluation.

The Rao‑Wilton‑Glisson (RWG) basis satisfies all these requirements and has become the standard for triangular surface meshes in electromagnetics.

---

## 2. RWG Basis Functions: Definition and Geometry

### 2.1 Geometric Intuition

Imagine a mesh surface composed of triangular patches. An RWG basis function is associated with **one interior edge** shared by exactly two triangles:

- **"Plus" triangle $T^+$**: current flows **toward** the shared edge from the opposite vertex.
- **"Minus" triangle $T^-$**: current flows **away from** the shared edge toward the opposite vertex.

The shared edge acts as a "bridge": the current that leaves triangle $T^+$ through the edge enters triangle $T^-$ through the same edge, guaranteeing continuity and charge conservation.

### 2.2 Mathematical Definition

For an interior edge $n$ of length $\ell_n$, denote the two triangles sharing the edge by $T_n^+$ and $T_n^-$, with areas $A_n^+$ and $A_n^-$, respectively. Let $\mathbf{r}_{+,\mathrm{opp}}$ and $\mathbf{r}_{-,\mathrm{opp}}$ be the vertices of $T_n^+$ and $T_n^-$ opposite the shared edge. The RWG basis function is defined as

```math
\mathbf{f}_n(\mathbf{r}) =
\begin{cases}
\displaystyle \frac{\ell_n}{2A_n^+}\,(\mathbf{r} - \mathbf{r}_{+,\mathrm{opp}}), & \mathbf{r} \in T_n^+, \\[8pt]
\displaystyle \frac{\ell_n}{2A_n^-}\,(\mathbf{r}_{-,\mathrm{opp}} - \mathbf{r}), & \mathbf{r} \in T_n^-, \\[8pt]
\mathbf{0}, & \text{otherwise}.
\end{cases}
```

The function is **piecewise linear** (affine) on each triangle and vanishes outside the two supporting triangles.

### 2.3 Geometric Interpretation

- **Edge length $\ell_n$**: determines the “strength” of the basis; a longer edge supports a larger current flow.
- **Triangle areas $A_n^\pm$**: normalize the current density; smaller triangles yield higher current density for the same total current.
- **Vectors $\mathbf{r} - \mathbf{r}_{+,\mathrm{opp}}$ and $\mathbf{r}_{-,\mathrm{opp}} - \mathbf{r}$**: produce a linear flow pattern that is radial with respect to the opposite vertex.

**Physical picture**: In $T_n^+$ the current flows radially toward the shared edge from the opposite vertex; in $T_n^-$ it flows radially away from the shared edge toward the opposite vertex. The scaling ensures that the normal component of current is continuous across the edge, i.e., the current leaving $T_n^+$ exactly equals the current entering $T_n^-$.

### 2.4 ASCII Diagram: RWG Basis Function Geometry

```
                        RWG Basis Function - Edge n
                        Shared edge length = ℓ_n
                      
                         T+ triangle          T- triangle
                            ▲                     ▼
                           ╱ ╲                   ╱ ╲
                          ╱   ╲                 ╱   ╲
                         ╱     ╲               ╱     ╲
                        ╱       ╲             ╱       ╲
                       ╱         ╲           ╱         ╲
                      ╱           ╲         ╱           ╲
                     ╱             ╲       ╱             ╲
                    ╱               ╲     ╱               ╲
                   ╱                 ╲   ╱                 ╲
                  ╱                   ╲ ╱                   ╲
                 ╱                     ●                     ╲
                ╱                     ╱ ╲                     ╲
               ╱                     ╱   ╲                     ╲
              ╱                     ╱     ╲                     ╲
             ╱                     ╱       ╲                     ╲
            ╱                     ╱         ╲                     ╲
           ╱                     ╱           ╲                     ╲
          ▼─────────────────────●─────────────●─────────────────────►
         p₁                    edge n         p₂                    Current
          (vertex)             (length ℓ_n)    (vertex)             flow direction
        
        
        Legend:
        ● = Edge endpoints (p₁, p₂)
        ▲ = Current flows TOWARD edge (T+ triangle)
        ▼ = Current flows AWAY from edge (T- triangle)
        ► = Linear current variation within triangle
        T+ area = A_n^+, T- area = A_n^-
```

### 2.5 Current Flow Visualization

**Triangle $T^+$ (plus triangle)**:
```
        Opposite vertex r_{+,opp}
                 ▲
                ╱ ╲
               ╱   ╲
              ╱     ╲   Current flows radially toward
             ╱       ╲   the shared edge from opposite vertex
            ╱         ╲
           ╱           ╲
          ╱             ╲
         ▼─────────────────►
      edge n            edge n
```

**Triangle $T^-$ (minus triangle)**:
```
        Opposite vertex r_{-,opp}
                 ▼
                ╱ ╲
               ╱   ╲
              ╱     ╲   Current flows radially away from
             ╱       ╲   the shared edge toward opposite vertex
            ╱         ╲
           ╱           ╲
          ╱             ╲
         ►─────────────────▲
      edge n            edge n
```

### 2.6 Current Continuity at the Shared Edge

```
    T+ triangle        |        T- triangle
                       |
        ▲              |              ▼
       ╱ ╲             |             ╱ ╲
      ╱   ╲            |            ╱   ╲
     ╱     ╲           |           ╱     ╲
    ╱       ╲          |          ╱       ╲
   ╱         ╲         |         ╱         ╲
  ╱           ╲        |        ╱           ╲
 ╱             ╲       |       ╱             ╲
╱               ╲      |      ╱               ╲
▼───────────────►│◄─────────────►───────────────▲
    edge n       │       edge n
                 │
    Current OUT  │  Current IN
    of T+        │  to T-
    equals       │  equals
    Current IN   │  Current OUT
    to T-        │  of T+
```

**Key insight**: The same current that leaves triangle $T^+$ through edge $n$ enters triangle $T^-$ through the same edge—ensuring charge conservation and avoiding unphysical charge accumulation.

### 2.7 Concrete Numerical Example

Consider an interior edge with:
- Length $\ell_n = 0.01\ \text{m}$,
- Triangle areas $A_n^+ = 6\times10^{-5}\ \text{m}^2$, $A_n^- = 8\times10^{-5}\ \text{m}^2$.

The scaling factors become
```math
\frac{\ell_n}{2A_n^+} \approx 83.3\ \text{m}^{-1}, \qquad
\frac{\ell_n}{2A_n^-} \approx 62.5\ \text{m}^{-1}.
```
The larger factor in the smaller triangle compensates for its reduced area, so that the total current flowing through the edge is the same from both sides.

---

## 3. Mathematical Properties of RWG Functions

### 3.1 Surface Divergence (Charge Density)

Because $\mathbf{f}_n$ is affine on each triangle, its surface divergence is **constant** on each support triangle:

```math
\nabla_s \cdot \mathbf{f}_n(\mathbf{r}) =
\begin{cases}
\displaystyle \frac{\ell_n}{A_n^+}, & \mathbf{r} \in T_n^+, \\[8pt]
\displaystyle -\frac{\ell_n}{A_n^-}, & \mathbf{r} \in T_n^-, \\[8pt]
0, & \text{otherwise}.
\end{cases}
```

**Derivation**: On $T_n^+$, write $\mathbf{f}_n(\mathbf{r}) = \frac{\ell_n}{2A_n^+}(\mathbf{r} - \mathbf{r}_{+,\mathrm{opp}})$. In local triangle coordinates, this is a linear function $\mathbf{f}_n = \mathbf{a} + \mathbf{B}\mathbf{r}$ with constant gradient $\mathbf{B}$. The surface divergence equals $\operatorname{tr}(\mathbf{B})$, which evaluates to $\ell_n/A_n^+$. The negative sign for $T_n^-$ arises from the reversed flow direction.

**Physical interpretation**: The surface divergence represents the **charge density** associated with the basis function via the continuity equation $\nabla_s \cdot \mathbf{J} = -i\omega \rho$. The constant divergence on each triangle greatly simplifies the scalar‑potential term in the EFIE assembly.

**Charge density connection**: The continuity equation for time‑harmonic fields links surface current divergence to surface charge density: $\nabla_s \cdot \mathbf{J} = -i\omega \rho_s$, where $\rho_s$ has units C/m². For an RWG basis function $\mathbf{f}_n$, the divergence is constant on each triangle, implying a piecewise‑constant charge density. On $T_n^+$, $\rho_s^+ = \frac{i}{\omega} \nabla_s\cdot\mathbf{f}_n = \frac{i}{\omega}\frac{\ell_n}{A_n^+}$; on $T_n^-$, $\rho_s^- = \frac{i}{\omega} \nabla_s\cdot\mathbf{f}_n = -\frac{i}{\omega}\frac{\ell_n}{A_n^-}$. The total charge on $T_n^+$ is $Q^+ = \rho_s^+ A_n^+ = \frac{i}{\omega}\ell_n$, and on $T_n^-$ is $Q^- = \rho_s^- A_n^- = -\frac{i}{\omega}\ell_n$. These charges sum to zero, reflecting local charge conservation: the current flowing out of $T_n^+$ through the shared edge exactly balances the current flowing into $T_n^-$.

### 3.2 Current Continuity (Divergence Conformity)

A key property of RWG functions is that they are **divergence‑conforming**: the normal component of $\mathbf{f}_n$ is continuous across the shared edge. Mathematically, for any point $\mathbf{r}$ on the interior edge,

```math
\mathbf{f}_n^+(\mathbf{r}) \cdot \hat{\mathbf{n}}^+ = -\mathbf{f}_n^-(\mathbf{r}) \cdot \hat{\mathbf{n}}^-,
```

where $\hat{\mathbf{n}}^+$ and $\hat{\mathbf{n}}^-$ are the outward unit normals of $T_n^+$ and $T_n^-$ on the edge. This ensures that no net charge accumulates on the edge, satisfying charge conservation in the discrete sense.

**Proof**: The definition of $\mathbf{f}_n$ guarantees that the normal component varies linearly along the edge and matches at the endpoints; the scaling factors $\ell_n/(2A_n^\pm)$ are chosen precisely to enforce this continuity.

### 3.3 Linear Independence and Completeness

The set $\{\mathbf{f}_n\}_{n=1}^N$ associated with all interior edges of a triangular mesh is linearly independent (provided the mesh is connected and has no degenerate edges). Moreover, any piecewise‑linear tangential vector field on the mesh that is continuous across edges can be expressed as a linear combination of RWG functions. This makes the RWG basis **complete** for representing surface currents at the lowest‑order (linear) approximation.

### 3.4 Relationship to Edge Elements

RWG functions are the two‑dimensional surface analogue of the **edge elements** (Nédélec elements) used in finite element methods. Each basis is associated with an edge rather than a node, reflecting the fact that currents flow along edges, not through vertices.

### 3.5 Practical Implications of Constant Divergence

The affine nature of RWG functions leads to a **constant surface divergence** on each support triangle, as derived in Section 3.1. This simple mathematical property has profound practical consequences for EFIE assembly:

1. **Efficient scalar‑potential term**: Because $\nabla_s\cdot\mathbf{f}_n$ is constant on each triangle, the scalar‑potential contribution $S_{mn}$ in the mixed‑potential EFIE can be computed with minimal arithmetic. The divergence values need only be evaluated once per triangle and can be reused at every quadrature point.

2. **Charge‑density interpretation**: The constant divergence directly gives the **charge density** associated with each basis function via the continuity equation $\nabla_s\cdot\mathbf{J} = -i\omega\rho$. On $T_n^+$ the charge density is $+\frac{i}{\omega}\frac{\ell_n}{A_n^+}$; on $T_n^-$ it is $-\frac{i}{\omega}\frac{\ell_n}{A_n^-}$. The total charge on the two triangles sums to zero, reflecting local charge conservation.

3. **Simplified integration**: When assembling the scalar‑potential term
   ```math
   S_{mn} = \iint (\nabla_s\cdot\mathbf{f}_m)(\nabla_s'\cdot\mathbf{f}_n)\,G(\mathbf{r},\mathbf{r}')\,dS\,dS',
   ```
   the divergence factors can be pulled outside the inner integral, reducing the double integral to a product of constants times a purely geometric integral. This is exploited in the code where `div_rwg` is precomputed and stored.

4. **Verification tool**: The constant‑divergence property provides a strong consistency check for RWG construction. For any basis index $n$,
   ```math
   A_n^+ \times (\nabla_s\cdot\mathbf{f}_n \text{ on } T_n^+) = \ell_n, \qquad
   A_n^- \times (\nabla_s\cdot\mathbf{f}_n \text{ on } T_n^-) = -\ell_n.
   ```
   These equalities are automatically verified in the package's test suite and can be used to debug custom meshes.

The implementation in `src/basis/RWG.jl` provides the function `div_rwg(rwg, n, tidx)` that returns the constant divergence of basis $n$ on triangle `tidx`. This function is called repeatedly during matrix assembly without recomputing geometric quantities.

---

## 4. Discrete EFIE Block Structure

With the RWG basis functions defined, we can now translate the continuous mixed‑potential EFIE into a finite linear system $\mathbf{Z}\mathbf{I}=\mathbf{V}$. This section details how each matrix entry $Z_{mn}$ is assembled, how the assembly loops are organized, and where the corresponding code resides.

### 4.1 Matrix Elements from the Mixed‑Potential Form

Recall from Chapter 2 the mixed‑potential EFIE operator acting on a surface current $\mathbf{J}$:

```math
\mathcal{T}[\mathbf{J}](\mathbf{r}) = -i\omega\mu_0 \int_\Gamma \mathbf{J}(\mathbf{r}')G(\mathbf{r},\mathbf{r}')\,dS'
+ \frac{i}{\omega\varepsilon_0}\nabla_s\int_\Gamma (\nabla_s'\cdot\mathbf{J}(\mathbf{r}'))G(\mathbf{r},\mathbf{r}')\,dS'.
```

Applying Galerkin testing with RWG basis functions $\mathbf{f}_m$ and expanding $\mathbf{J}\approx\sum_n I_n\mathbf{f}_n$ yields the matrix element

```math
Z_{mn}^{\mathrm{EFIE}} = -i\omega\mu_0\Bigl[V_{mn} - \frac{1}{k^2}S_{mn}\Bigr],
```

where the **vector‑potential** and **scalar‑potential** contributions are

```math
\begin{aligned}
V_{mn} &= \iint_\Gamma \mathbf{f}_m(\mathbf{r})\cdot\mathbf{f}_n(\mathbf{r}')\,G(\mathbf{r},\mathbf{r}')\,dS\,dS', \\[4pt]
S_{mn} &= \iint_\Gamma \bigl(\nabla_s\cdot\mathbf{f}_m(\mathbf{r})\bigr)\bigl(\nabla_s'\cdot\mathbf{f}_n(\mathbf{r}')\bigr)\,G(\mathbf{r},\mathbf{r}')\,dS\,dS'.
\end{aligned}
```

The factor $1/k^2 = 1/(\omega^2\mu_0\varepsilon_0)$ converts the scalar‑potential term to the same units as the vector‑potential term. The minus sign between the two terms arises from the integration‑by‑parts step described in Chapter 2.

### 4.2 Assembly Loop Structure

Because each RWG basis function is supported on exactly two triangles, the double‑surface integrals reduce to sums over at most four triangle pairs. For a given test‑source pair $(m,n)$:

- Let $T_m^+, T_m^-$ be the two triangles supporting $\mathbf{f}_m$.
- Let $T_n^+, T_n^-$ be the two triangles supporting $\mathbf{f}_n$.

The matrix entry $Z_{mn}$ is assembled as

```math
Z_{mn} = \sum_{p\in\{+,-\}}\sum_{q\in\{+,-\}}
\Bigl[ -i\omega\mu_0\,V_{mn}^{pq} + \frac{i}{\omega\varepsilon_0}\,S_{mn}^{pq} \Bigr],
```

where $V_{mn}^{pq}$ and $S_{mn}^{pq}$ denote the integrals restricted to triangle pair $(T_m^p, T_n^q)$. In practice, the assembly loops over **all interior edges** (test basis indices $m$) and for each $m$ over all interior edges (source basis indices $n$), accumulating contributions from the relevant triangle pairs.

#### Pseudo‑code for the core assembly loop

```
for m in 1:Nedges
    (T_m_plus, T_m_minus) = support_triangles(rwg, m)
    for n in 1:Nedges
        (T_n_plus, T_n_minus) = support_triangles(rwg, n)
        Z_temp = 0
        for p in (T_m_plus, T_m_minus)
            for q in (T_n_plus, T_n_minus)
                # Evaluate quadrature points on triangle p and q
                # Compute vector part: f_m(r) · f_n(r') * G(r,r')
                # Compute scalar part: div_m * div_n * G(r,r')
                # Accumulate with proper weights
                Z_temp += contribution
            end
        end
        Z[m,n] = Z_temp
    end
end
```

Although this naïve double loop has $O(N^2)$ complexity, the implementation in `DiffMoM.jl` keeps the same asymptotic cost but reduces overhead by caching triangle quadrature points/areas and per-basis RWG values.

### 4.3 Mapping to the Code: `vec_part - scl_part`

In the file `src/assembly/EFIE.jl`, the matrix assembly explicitly separates vector and scalar contributions. The core update is

```julia
vec_part = dot(fm, fn) * G
scl_part = dvm * dvn * G / (cache.k^2)
val += (vec_part - scl_part) * weight
```

where
- `vec_part` is the vector-potential integrand contribution inside the bracketed term,
- `scl_part` is the scalar-potential integrand contribution inside the bracketed term.

The subtraction reflects the minus sign in the formula $Z_{mn}= -i\omega\mu_0[V_{mn}-k^{-2}S_{mn}]$. The factor $-i\omega\mu_0$ is factored out of both terms, so the integrand computed at each quadrature pair is precisely $\mathbf{f}_m\cdot\mathbf{f}_n\,G - k^{-2}(\nabla_s\cdot\mathbf{f}_m)(\nabla_s'\cdot\mathbf{f}_n)G$.

#### Code location and function calls

- **Entry point**: `assemble_Z_efie(mesh, rwg, k; quad_order=3)` in `src/assembly/EFIE.jl`.
- **Quadrature evaluation**: `tri_quad_rule` and `tri_quad_points` from `src/basis/Quadrature.jl`.
- **RWG evaluation**: `eval_rwg(rwg, n, r, tidx)` and `div_rwg(rwg, n, tidx)` from `src/basis/RWG.jl`.
- **Green’s function**: `greens(r, rp, k)` defined in `src/basis/Greens.jl`.

The assembly is implemented with explicit nested loops (`m`, `n`, support triangles, quadrature points) plus precomputed cache data. There is no symmetry-based half-matrix reduction in the dense assembler; all entries are evaluated directly.

### 4.4 Complexity and Sparsity Considerations

Despite the local support of RWG functions, the EFIE operator is **dense**: every basis function couples to every other basis function through the Green’s function $G(\mathbf{r},\mathbf{r}')$, which decays only as $1/R$. Consequently, the impedance matrix $\mathbf{Z}$ is fully populated, requiring $O(N^2)$ storage and $O(N^2)$ time for direct assembly.

However, the **local support** of RWG basis functions still provides important benefits:

1. **Compact quadrature loops**: Each matrix entry involves at most four triangle pairs, keeping the inner‑loop geometry simple.
2. **Efficient geometric queries**: The mesh connectivity needed for assembly is limited to the two triangles per basis function.
3. **Compatible with fast methods**: The same RWG discretization can be used with the package’s fast/operator-based pathways (e.g., matrix-free and MLFMA components) when dense assembly is too expensive.

For moderate problems ($N \lesssim 10^4$), the direct $O(N^2)$ assembly is feasible and is the method implemented in the current version of `DiffMoM.jl`.

---

## 5. Numerical Integration: Quadrature and Triangle Mapping

Accurate evaluation of the double‑surface integrals $V_{mn}$ and $S_{mn}$ is essential for the convergence of the MoM solution. Because the integrands are smooth except when triangles are close together or touching, Gaussian quadrature on triangles provides an efficient and accurate integration scheme. This section explains the reference‑triangle quadrature used in `DiffMoM.jl`, the Jacobian factor $2A_T$, and how the quadrature data is organized in the code.

### 5.1 Reference Triangle and Gaussian Quadrature

The standard reference triangle $\hat{T}$ is defined by vertices $(0,0)$, $(1,0)$, and $(0,1)$ in a two‑dimensional parameter space $(\xi,\eta)$. Its area is $1/2$. A $N_q$‑point Gaussian quadrature rule on $\hat{T}$ consists of weights $w_q$ and points $(\xi_q,\eta_q)$ chosen so that

```math
\int_{\hat{T}} f(\xi,\eta)\,d\xi d\eta \approx \sum_{q=1}^{N_q} w_q f(\xi_q,\eta_q)
```

integrates polynomials of a certain degree exactly. The package provides several rules with different numbers of points and degrees of precision. Note that `tri_quad_rule` requires an explicit `order` argument (e.g., `tri_quad_rule(3)`). The default in `assemble_Z_efie` is `quad_order=3`, which gives a 3-point rule. Higher-order rules (e.g., `tri_quad_rule(7)` for a 7-point rule) integrate polynomials up to higher degrees exactly, which can improve accuracy for EFIE integrals involving smooth basis functions and the Green's function. Supported orders are 1, 3, 4, and 7.

### 5.2 Mapping to Physical Triangles

Each physical triangle $T$ of the mesh is defined by three vertices $\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3 \in \mathbb{R}^3$. A linear mapping $\mathbf{r}(\xi,\eta) = \mathbf{v}_1 + \xi(\mathbf{v}_2-\mathbf{v}_1) + \eta(\mathbf{v}_3-\mathbf{v}_1)$ transforms the reference triangle to $T$. The Jacobian determinant of this affine mapping is constant and equals twice the physical triangle area:

```math
J = \left\| \frac{\partial\mathbf{r}}{\partial\xi} \times \frac{\partial\mathbf{r}}{\partial\eta} \right\| = 2A_T.
```

Consequently, an integral over the physical triangle becomes

```math
\int_T f(\mathbf{r})\,dS = \int_{\hat{T}} f(\mathbf{r}(\xi,\eta))\,J\,d\xi d\eta
\approx \sum_{q=1}^{N_q} w_q\, f(\mathbf{r}(\xi_q,\eta_q)) \, (2A_T).
```

The factor $2A_T$ appears **outside** the sum because the Jacobian is constant over the triangle. This factor is a frequent source of hidden scaling bugs; forgetting it leads to results that are off by a factor proportional to triangle area.

### 5.3 Implementation in `src/basis/Quadrature.jl`

The quadrature module provides two key functions:

- `tri_quad_rule(order)` → returns `(xi, w)` where `xi` is a vector of
  reference-triangle points `(ξ,η)` and `w` is the corresponding weight vector.
- `tri_quad_points(mesh, t, xi)` → maps the reference‑triangle quadrature points to physical coordinates for a given triangle.

**Example usage**:

```julia
using DiffMoM

mesh = make_rect_plate(0.1, 0.1, 2, 2)
xi, w = tri_quad_rule(7)             # order-7 rule (7 points)
points_phys = tri_quad_points(mesh, 1, xi)
```

During EFIE assembly, `tri_quad_points` is called once per triangle and the results are cached to avoid repeated computation. The weights and Jacobian factor $2A_T$ are combined into a single weight vector `w_q * (2A_T)` for efficiency.

### 5.4 Accuracy and Convergence Considerations

The choice of quadrature rule affects both accuracy and computational cost. For well‑separated triangle pairs, the integrand is smooth and a moderate‑order rule (e.g., 7 points) yields high accuracy. When the source and test triangles coincide, the Green’s kernel is singular. In `DiffMoM.jl`, that self-cell singularity is handled by analytical extraction (`src/assembly/SingularIntegrals.jl`) rather than by increasing quadrature order alone.

A useful rule of thumb: the quadrature rule should integrate polynomials of degree $2p$ exactly, where $p$ is the polynomial degree of the basis functions. Since RWG functions are linear (affine) on each triangle, $p=1$, so a rule that integrates degree‑2 polynomials exactly is sufficient for the vector part $\mathbf{f}_m\cdot\mathbf{f}_n$. However, the Green’s function adds complexity, and a higher‑order rule is often used in practice to maintain accuracy across a range of distances.

---

## 6. Mesh Quality and Preconditions for RWG Construction

The correctness of RWG basis functions—and hence the accuracy of the MoM solution—depends critically on the quality of the input triangular mesh. Invalid topology (non‑manifold edges, inconsistent orientations) or degenerate geometry (zero‑area triangles, extremely skewed elements) can lead to ill‑defined basis functions, singular matrices, or subtle convergence failures. This section describes the mesh‑quality checks performed by `DiffMoM.jl` and how to use them in practice.

### 6.1 Why Mesh Quality Matters

RWG functions are built on the assumption that each interior edge is shared by **exactly two triangles** and that those triangles are non‑degenerate (positive area). Violations of these assumptions cause:

- **Non‑manifold edges**: an edge belonging to three or more triangles cannot be assigned a unique “plus” and “minus” triangle, breaking the RWG definition.
- **Degenerate triangles**: triangles with zero (or nearly zero) area produce infinite scaling factors $\ell_n/(2A_n^\pm)$, leading to numerical overflow and ill‑conditioned matrices.
- **Orientation mismatches**: the outward normals of the two triangles sharing an edge must be consistently oriented; otherwise the sign conventions for current flow become ambiguous.
- **Out‑of‑range indices**: references to non‑existent vertices or triangles indicate corrupted mesh data.

The package includes a comprehensive mesh‑quality precheck that catches these issues before RWG construction, providing clear error messages that help diagnose the problem.

### 6.2 Common Mesh Defects Detected

The function `mesh_quality_report(mesh)` scans the mesh and returns a structured report containing the following checks:

| Check | Description | Typical cause |
|-------|-------------|---------------|
| `n_invalid_triangles` + `invalid_triangles` | Triangle has out-of-range indices or repeated vertices. | Off-by-one indexing, corrupt connectivity. |
| `n_degenerate_triangles` + `degenerate_triangles` | Triangle area is below tolerance (`area_tol_abs`). | Collapsed/near-collapsed elements. |
| `n_nonmanifold_edges` | An edge has more than two incident triangles. | Non-manifold topology, T-junction style defects. |
| `n_orientation_conflicts` | Interior-edge orientation is inconsistent. | Mixed triangle winding conventions. |
| `n_boundary_edges` | Edges with one incident triangle (open boundary). | Open surfaces or missing facets. |
| `n_edges_total`, `n_interior_edges` | Topology summary counts. | Mesh-connectivity diagnostics. |

The report also includes `area_tol_abs`, which is the absolute area threshold derived from `area_tol_rel` and mesh scale.

### 6.3 Using `mesh_quality_report` and `mesh_quality_ok`

To inspect a mesh before building RWG functions, call:

```julia
using DiffMoM

mesh = make_rect_plate(0.1, 0.1, 10, 10)   # example mesh
report = mesh_quality_report(mesh)
println(report)
```

`mesh_quality_report` returns a named tuple of counts and index lists. To obtain a simple pass/fail verdict, use

```julia
ok = mesh_quality_ok(report; allow_boundary=true, require_closed=false)
```

The keyword arguments control whether boundary edges (edges belonging to only one triangle) are allowed (`allow_boundary=true` for open surfaces) and whether the mesh must be closed (`require_closed=false` for open surfaces, `true` for closed scatterers).

### 6.4 Automatic Precheck in `build_rwg`

By default, `build_rwg(mesh; precheck=true)` calls `assert_mesh_quality`, which internally computes `mesh_quality_report` and throws an informative error if hard checks fail. This prevents silent failures later during matrix assembly. If you are confident that your mesh is valid (e.g., from a trusted meshing tool), you can disable the precheck with `precheck=false` for a slight speed gain.

**Code location**: The mesh‑quality routines are implemented in `src/geometry/Mesh.jl` (notably `mesh_quality_report`, `mesh_quality_ok`, and `assert_mesh_quality`). The RWG constructor `build_rwg` in `src/basis/RWG.jl` calls these checks when `precheck=true`.

---

## 7. Hands‑On Example: Inspecting RWG Data Structures

To solidify the concepts presented in this chapter, we walk through a complete Julia session that creates a simple mesh, builds the RWG basis functions, examines their geometric properties, and verifies key mathematical relations. This example demonstrates the inspection tools provided by `DiffMoM.jl` and serves as a template for debugging custom meshes.

### 7.1 Creating a Simple Mesh

We start by generating a rectangular plate discretized into triangles using the built‑in utility `make_rect_plate`.

```julia
using DiffMoM

# Create a 0.1 m × 0.1 m plate meshed with 4×4 subdivisions
mesh = make_rect_plate(0.1, 0.1, 4, 4)

println("Mesh vertices: ", nvertices(mesh))
println("Mesh triangles: ", ntriangles(mesh))
```

The resulting mesh contains `nvertices(mesh)` vertices and `ntriangles(mesh)` triangles. Because the plate is open (a single surface), all edges on the boundary belong to only one triangle, while interior edges are shared by two triangles.

### 7.2 Building RWG Basis Functions

With the mesh ready, we construct the RWG basis functions:

```julia
rwg = build_rwg(mesh; precheck=true)

println("Number of RWG basis functions (interior edges): ", rwg.nedges)
```

The object `rwg` stores all geometric data needed to evaluate basis functions and their divergences. The `precheck=true` flag ensures that the mesh passes the quality checks described in Section 6 before proceeding.

### 7.3 Examining Basis Properties

Let’s inspect the first few basis functions. The function `basis_triangles(rwg, n)` returns the indices of the “plus” and “minus” triangles supporting basis $n$.

```julia
for n in 1:min(3, rwg.nedges)
    tp, tm = basis_triangles(rwg, n)
    println("Basis $n: plus triangle = $tp, minus triangle = $tm")
end
```

To see the actual geometry, we can retrieve the edge length and triangle areas:

```julia
n = 1
ℓ = rwg.len[n]
A_plus = triangle_area(mesh, rwg.tplus[n])
A_minus = triangle_area(mesh, rwg.tminus[n])

println("Edge length ℓ_$n = $ℓ m")
println("Area of T⁺ = $A_plus m², Area of T⁻ = $A_minus m²")
```

### 7.4 Verifying Consistency Relations

The defining property of RWG functions is that the normal component of current is continuous across the shared edge. This translates to the relation

```math
A_n^+ \times (\nabla_s\cdot\mathbf{f}_n \text{ on } T_n^+) = \ell_n, \qquad
A_n^- \times (\nabla_s\cdot\mathbf{f}_n \text{ on } T_n^-) = -\ell_n.
```

We can check this numerically using `div_rwg`:

```julia
div_plus = div_rwg(rwg, n, rwg.tplus[n])
div_minus = div_rwg(rwg, n, rwg.tminus[n])

println("∇·f_$n on T⁺ = $div_plus")
println("∇·f_$n on T⁻ = $div_minus")
println("A⁺ × div_plus = $(A_plus * div_plus) (should be ≈ $ℓ)")
println("A⁻ × div_minus = $(A_minus * div_minus) (should be ≈ -$ℓ)")
```

The two products should equal $+\ell_n$ and $-\ell_n$ up to rounding error (typically $\sim 10^{-15}$ for double‑precision arithmetic). If they differ significantly, the mesh likely contains geometric inconsistencies.

### 7.5 Visualizing Current Flow (Optional)

`DiffMoM.jl` includes mesh-level visualization utilities in
`src/postprocessing/Visualization.jl`, and RWG values can also be inspected directly. The
following snippet shows how to extract the current direction at the centroid of
each support triangle:

```julia
using LinearAlgebra

# Evaluate f_n at the centroid of its plus triangle
centroid_plus = (Vec3(mesh.xyz[:, mesh.tri[1, tp]]) +
                 Vec3(mesh.xyz[:, mesh.tri[2, tp]]) +
                 Vec3(mesh.xyz[:, mesh.tri[3, tp]])) / 3
f_plus = eval_rwg(rwg, n, centroid_plus, tp)

println("f_$n at centroid of T⁺ = $f_plus (direction: $(normalize(f_plus)))")
```

The vector `f_plus` points radially away from the shared edge toward the opposite vertex, as described in Section 2.

### 7.6 Complete Inspection Script

Putting everything together, here is a complete script that performs all the checks and prints a summary:

```julia
using DiffMoM
using LinearAlgebra

function inspect_rwg(mesh, max_to_show=5)
    rwg = build_rwg(mesh; precheck=true)
    println("RWG basis count: ", rwg.nedges)
    println("\nChecking consistency for the first ", max_to_show, " bases:")
    for n in 1:min(max_to_show, rwg.nedges)
        tp = rwg.tplus[n]
        tm = rwg.tminus[n]
        ℓ = rwg.len[n]
        A⁺ = triangle_area(mesh, tp)
        A⁻ = triangle_area(mesh, tm)
        div⁺ = div_rwg(rwg, n, tp)
        div⁻ = div_rwg(rwg, n, tm)
        err⁺ = abs(A⁺ * div⁺ - ℓ)
        err⁻ = abs(A⁻ * div⁻ + ℓ)
        println("  Basis $n: ℓ=$ℓ, err⁺=$(err⁺), err⁻=$(err⁻) ",
                err⁺ < 1e-12 && err⁻ < 1e-12 ? "✓" : "✗")
    end
    return rwg
end

mesh = make_rect_plate(0.1, 0.1, 4, 4)
rwg = inspect_rwg(mesh, 5)
```

Running this script should produce a list of basis functions with negligible errors, confirming that the RWG construction is geometrically consistent.

---

## 8. Code Mapping and Implementation Details

This section provides a concise roadmap to the source files that implement the RWG discretization and EFIE assembly. Knowing where each piece of functionality resides is essential for extending the code, debugging, or adapting it to new integral equations.

### 8.1 Overview of Source Files

| File | Purpose | Key contents |
|------|---------|--------------|
| `src/basis/RWG.jl` | Construction and evaluation of RWG basis functions. | `RWGData`, `build_rwg`, `eval_rwg`, `div_rwg`, `basis_triangles`. |
| `src/geometry/Mesh.jl` | Mesh types, geometry helpers, and quality checks. | `TriMesh`, `make_rect_plate`, `triangle_area`, `mesh_quality_report`, `assert_mesh_quality`. |
| `src/basis/Quadrature.jl` | Gaussian quadrature rules on triangles. | `tri_quad_rule`, `tri_quad_points`. |
| `src/basis/Greens.jl` | Free‑space Green’s function utilities. | `greens`, `greens_smooth`, `grad_greens`. |
| `src/assembly/SingularIntegrals.jl` | Self-cell singularity extraction. | `analytical_integral_1overR`, `self_cell_contribution`. |
| `src/assembly/EFIE.jl` | Assembly of the EFIE operator matrix. | `assemble_Z_efie`. |
| `src/assembly/Impedance.jl` | Impedance-term assembly and derivatives. | `precompute_patch_mass`, `assemble_Z_impedance`, `assemble_dZ_dtheta`. |
| `src/solver/Solve.jl` | Linear solves and conditioned systems. | `solve_forward`, `solve_system`, `assemble_full_Z`, `prepare_conditioned_system`. |

### 8.2 Key Data Structures

**`RWGData`** (defined in `src/Types.jl` / constructed in `src/basis/RWG.jl`):
- `tplus::Vector{Int}` – indices of the “plus” triangle for each basis.
- `tminus::Vector{Int}` – indices of the “minus” triangle for each basis.
- `vplus_opp::Vector{Int}` – vertex opposite the shared edge in the plus triangle.
- `vminus_opp::Vector{Int}` – vertex opposite the shared edge in the minus triangle.
- `evert::Matrix{Int}` – edge-vertex indices for each basis.
- `nedges::Int` – number of interior edges (equal to number of RWG basis functions).
- `mesh::TriMesh` – reference to the underlying mesh.
- `len`, `area_plus`, `area_minus` – geometric factors used by `eval_rwg` and `div_rwg`.

**`TriMesh`** (defined in `src/Types.jl`):
- `xyz::Matrix{Float64}` – `(3, Nv)` array of vertex coordinates.
- `tri::Matrix{Int}` – `(3, Nt)` array of triangle vertex indices.
- `nvertices(mesh)` / `ntriangles(mesh)` – helper accessors.

### 8.3 Important Functions for Evaluation and Assembly

| Function | File | Description |
|----------|------|-------------|
| `eval_rwg(rwg, n, r, tidx)` | `src/basis/RWG.jl` | Evaluate $\mathbf{f}_n(\mathbf{r})$ on triangle `tidx` at point `r`. |
| `div_rwg(rwg, n, tidx)` | `src/basis/RWG.jl` | Return $\nabla_s\cdot\mathbf{f}_n$ on triangle `tidx` (constant). |
| `rwg.len[n]` | `src/basis/RWG.jl` | Length $\ell_n$ of the edge associated with basis $n$. |
| `triangle_area(mesh, tidx)` | `src/geometry/Mesh.jl` | Area of triangle `tidx`. |
| `basis_triangles(rwg, n)` | `src/basis/RWG.jl` | Return `(plus_triangle, minus_triangle)` for basis $n$. |
| `tri_quad_points(mesh, t, xi)` | `src/basis/Quadrature.jl` | Map quadrature points from reference to physical triangle. |
| `assemble_Z_efie(mesh, rwg, k; quad_order=3)` | `src/assembly/EFIE.jl` | Assemble the full EFIE impedance matrix. |
| `greens(r, rp, k)` | `src/basis/Greens.jl` | Compute $e^{-ikR}/(4\pi R)$. |

### 8.4 How to Extend the Code

If you wish to implement a different integral equation (e.g., Magnetic Field Integral Equation, MFIE) or a higher‑order basis function, the following steps are recommended:

1. **Add new basis functions** in a new source file under `src/` (for example, a future `HigherOrderBasis.jl`) following the pattern of `src/basis/RWG.jl`. Provide evaluation and divergence routines.
2. **Create a new assembly routine** in a new file under `src/` (for example, a future `MFIE.jl`) that mirrors the structure of `src/assembly/EFIE.jl` but implements the desired operator.
3. **Extend the mesh quality checks** if the new basis requires additional geometric constraints (e.g., curved elements).
4. **Integrate with the solve path** by adding a wrapper that assembles your new operator and calls `solve_forward`/`solve_system` consistently.

The modular design of `DiffMoM.jl` separates geometry, basis functions, quadrature, and operator assembly, making such extensions straightforward.

---

## 9. Exercises and Problems

The following exercises reinforce the key concepts of this chapter, ranging from mathematical derivations to hands‑on coding tasks. Working through them will deepen your understanding of RWG basis functions and their role in MoM discretization.

### 9.1 Conceptual Questions

1. **Current continuity**: Explain in physical terms why the normal component of an RWG basis function must be continuous across the shared edge. What would happen if this continuity were violated?
2. **Charge conservation**: Using the continuity equation $\nabla_s\cdot\mathbf{J} = -i\omega\rho$, show that the total charge associated with a single RWG basis function is zero. Relate this to the constant‑divergence property.
3. **Support size**: Why does each RWG basis function have exactly two supporting triangles? Could one design a basis function that spans three or more triangles while maintaining current continuity? What would be the advantages/disadvantages?
4. **Edge vs. node bases**: Contrast RWG (edge‑based) basis functions with node‑based scalar finite‑element bases. Why are edge‑based bases more natural for representing surface currents?

### 9.2 Derivation Tasks

5. **Galerkin discretization**: Starting from the tested residual equation
   ```math
   \langle \mathbf{f}_m, \mathcal{T}[\mathbf{J}] - Z_s\mathbf{J} \rangle = -\langle \mathbf{f}_m, \mathbf{E}_t^{\mathrm{inc}} \rangle,
   ```
   and the expansion $\mathbf{J} = \sum_n I_n \mathbf{f}_n$, derive the linear system $\sum_n Z_{mn} I_n = V_m$. Clearly state the definitions of $Z_{mn}$ and $V_m$.

6. **Constant divergence proof**: Prove that the surface divergence of an RWG function is constant on each support triangle. Hint: Write $\mathbf{f}_n(\mathbf{r}) = \mathbf{a} + \mathbf{B}\mathbf{r}$ on a triangle and compute $\nabla_s\cdot\mathbf{f}_n = \operatorname{tr}(\mathbf{B})$.

7. **Edge‑length relation**: Show that for an RWG basis function associated with edge $n$,
   ```math
   A_n^+ \times (\nabla_s\cdot\mathbf{f}_n \text{ on } T_n^+) = \ell_n, \qquad
   A_n^- \times (\nabla_s\cdot\mathbf{f}_n \text{ on } T_n^-) = -\ell_n.
   ```
   Use the geometric definition of $\mathbf{f}_n$ and the fact that the divergence is constant.

8. **Matrix symmetry**: Under what conditions is the EFIE impedance matrix $\mathbf{Z}$ symmetric? Does the inclusion of an impedance boundary condition $Z_s$ affect symmetry? Explain.

### 9.3 Coding and Verification

9. **Mesh inspection script**: Write a Julia script that loads a mesh from a file (using `read_obj_mesh`, or create one with `make_rect_plate`), runs `mesh_quality_report`, and prints a summary of any defects.

10. **RWG consistency check**: Implement the function `verify_rwg_consistency(rwg)` that loops over all basis functions and verifies the edge‑length relations from Exercise 7. The function should return `true` if all checks pass within a tolerance of $10^{-12}$ and `false` otherwise.

11. **Quadrature convergence**: Choose a simple analytic function $f(\mathbf{r}) = x^2 + y^2$ defined on a triangle. Compute its integral over the triangle using `tri_quad_rule(1)`, `tri_quad_rule(3)`, `tri_quad_rule(4)`, and `tri_quad_rule(7)`. Compare with the exact integral obtained from analytic geometry and plot the error vs. number of quadrature points.

12. **Basis function plotter**: Using a plotting package of your choice (e.g., `Plots.jl` with `pyplot` backend), visualize the vector field of an RWG basis function on its two supporting triangles. Represent the current direction with arrows and use a color map to indicate the magnitude.

### 9.4 Advanced Challenges

13. **Alternative basis functions**: Implement a “rooftop” basis function defined on a rectangular patch (two triangles forming a rectangle). Define the basis as a linear function on each triangle that ensures continuity across the shared edge. Compare its properties with the RWG basis.

14. **Singular integration**: Investigate the accuracy of Gaussian quadrature for the double‑surface integral $V_{mn}$ when triangles $T_m$ and $T_n$ are close together. Write a script that computes $V_{mn}$ for a fixed pair of triangles as their separation distance $d$ decreases from $0.1\lambda$ to $10^{-6}\lambda$. Plot the relative error (compared to a reference high‑order quadrature) vs. $d$ and discuss the onset of the “singular integration” problem.

15. **Differentiable assembly**: The package provides derivative-verification utilities (`complex_step_grad`, `fd_grad`) in `src/optimization/Verification.jl`. Write a small test that checks $\partial Z_{mn}/\partial k$ for selected entries against a centered finite-difference estimate.

---

## 10. Chapter Checklist

Before proceeding to Chapter 4, ensure you understand:

- [ ] The Galerkin discretization of the EFIE and the definition of the impedance matrix $Z_{mn}$ and excitation vector $V_m$.
- [ ] The geometric definition of an RWG basis function and the meaning of the “plus” and “minus” triangles.
- [ ] The proof that RWG functions are divergence‑conforming and the derivation of their constant surface divergence.
- [ ] How the mixed‑potential EFIE matrix element $Z_{mn}^{\mathrm{EFIE}}$ splits into vector‑potential ($V_{mn}$) and scalar‑potential ($S_{mn}$) contributions.
- [ ] The structure of the assembly loops: loops over test and source bases, then over their support triangles, then over quadrature points.
- [ ] The role of reference‑triangle quadrature and the origin of the Jacobian factor $2A_T$.
- [ ] The mesh‑quality checks performed before RWG construction and how to interpret `mesh_quality_report`.
- [ ] How to inspect RWG data structures using `basis_triangles`, `div_rwg`, `rwg.len[n]`, and `triangle_area`.
- [ ] Where to find the key implementation files: `src/basis/RWG.jl`, `src/geometry/Mesh.jl`, `src/basis/Quadrature.jl`, `src/assembly/EFIE.jl`.

If any items are unclear, review the relevant sections or consult the mathematical prerequisites appendix.

## 11. Further Reading

1. **Original RWG paper**:
   - Rao, S. M., Wilton, D. R., & Glisson, A. W. (1982). *Electromagnetic scattering by surfaces of arbitrary shape*. IEEE Transactions on Antennas and Propagation, 30(3), 409‑418. (The seminal paper introducing the RWG basis.)

2. **Comprehensive textbooks on MoM**:
   - Harrington, R. F. (1993). *Field Computation by Moment Methods*. IEEE Press. (Classic text covering MoM foundations and many basis functions.)
   - Peterson, A. F., Ray, S. L., & Mittra, R. (1998). *Computational Methods for Electromagnetics*. IEEE Press. (Includes detailed discussion of RWG functions and EFIE assembly.)
   - Chew, W. C., Jin, J. M., Michielssen, E., & Song, J. (2001). *Fast and Efficient Algorithms in Computational Electromagnetics*. Artech House. (Covers advanced topics including fast multipole acceleration.)

3. **Finite element edge elements**:
   - Nédélec, J. C. (1980). *Mixed finite elements in ℝ³*. Numerische Mathematik, 35(3), 315‑341. (The original paper on edge elements, conceptually related to RWG functions.)
   - Jin, J. M. (2014). *The Finite Element Method in Electromagnetics* (3rd ed.). Wiley. (Connects RWG bases to finite element edge elements.)

4. **Numerical integration on triangles**:
   - Dunavant, D. A. (1985). *High degree efficient symmetrical Gaussian quadrature rules for the triangle*. International Journal for Numerical Methods in Engineering, 21(6), 1129‑1148. (Source of many high‑order quadrature rules used in practice.)

5. **Mesh quality and generation**:
   - Shewchuk, J. R. (2002). *Delaunay refinement algorithms for triangular mesh generation*. Computational Geometry, 22(1‑3), 21‑74. (Foundational work on robust triangle mesh generation.)
   - Si, H. (2015). *TetGen, a Delaunay‑based quality tetrahedral mesh generator*. ACM Transactions on Mathematical Software, 41(2), 1‑36. (Includes surface meshing techniques.)

---

*Next: Chapter 4, “Singular Integration,” addresses the numerical challenges that arise when source and observation triangles are close together and presents specialized quadrature techniques to maintain accuracy.*
