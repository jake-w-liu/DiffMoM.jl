# EFIE and Boundary Conditions

## Purpose

This chapter provides a detailed derivation of the Electric Field Integral Equation (EFIE) in the mixed-potential form used throughout `DiffMoM.jl`. We connect each mathematical term to its physical interpretation and numerical implementation, with particular emphasis on boundary conditions for perfect electric conductors (PEC) and impedance surfaces.

After establishing the time-harmonic convention and Green's function, we derive the mixed-potential EFIE operator, explain the integration-by-parts step that separates vector and scalar contributions, and show how PEC and impedance boundary conditions translate to linear systems. The chapter concludes with implementation details, sign-convention checks, and practical examples.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive the PEC and impedance-sheet equations from tangential boundary conditions.
2. Map the continuous EFIE terms to the assembled matrix blocks.
3. Explain the physical meaning of the vector-potential and scalar-potential contributions in the mixed-potential form.
4. Understand why the impedance term enters with a negative sign in the system matrix.
5. Verify sign conventions by comparing the mathematical formulas with the actual Julia implementation.

---

## 1. Time Convention and Green's Function

### 1.1 Time-Harmonic Convention

`DiffMoM.jl` adopts the **$e^{+i\omega t}$** time-harmonic convention. This choice affects the sign of imaginary units in Maxwell's equations and the exponential argument of the Green's function. Under this convention:

```math
\frac{\partial}{\partial t} \rightarrow +i\omega, \qquad
\frac{\partial^2}{\partial t^2} \rightarrow -\omega^2.
```

All field quantities are understood as phasors: the physical time-domain field $\widetilde{\mathbf{E}}(\mathbf{r},t)$ is obtained via

```math
\widetilde{\mathbf{E}}(\mathbf{r},t) = \Re\left\{\mathbf{E}(\mathbf{r})\,e^{+i\omega t}\right\}.
```

### 1.2 Free-Space Scalar Green's Function

With the $e^{+i\omega t}$ convention, the outgoing-wave solution of the scalar Helmholtz equation

```math
(\nabla^2 + k^2)\,G(\mathbf{r},\mathbf{r}') = -\delta(\mathbf{r}-\mathbf{r}')
```

is

```math
G(\mathbf{r},\mathbf{r}') = \frac{e^{-ikR}}{4\pi R}, \qquad R = |\mathbf{r}-\mathbf{r}'|.
```

This Green's function automatically satisfies the Sommerfeld radiation condition and is implemented by the function `greens` in `src/basis/Greens.jl`.

### 1.3 Sign Consistency Check

A quick sanity check verifies the sign of the exponential. The Helmholtz operator $(\nabla^2 + k^2)$ with $k = \omega\sqrt{\mu_0\epsilon_0}$ and time derivative $\partial/\partial t \rightarrow +i\omega$ leads to the wave equation $(\nabla^2 + k^2)u = -\delta$. The outgoing fundamental solution must contain $e^{-ikR}$ to represent waves propagating radially outward. If a reference uses the opposite convention $e^{-i\omega t}$, the exponential flips to $e^{+ikR}$.

**Important:** When comparing formulas from different textbooks or codes, always check the time convention first. This package consistently uses $e^{+i\omega t}$ throughout.

---

## 2. Field Decomposition and Tangential Projection

### 2.1 Total Field Decomposition

As introduced in Chapter 1, the total electric field is decomposed into incident and scattered components:

```math
\mathbf{E}^{\mathrm{tot}}(\mathbf{r}) = \mathbf{E}^{\mathrm{inc}}(\mathbf{r}) + \mathbf{E}^{\mathrm{sca}}(\mathbf{r}), \qquad \mathbf{r} \in \Gamma
```

where $\Gamma$ denotes the surface of the scattering object. The incident field $\mathbf{E}^{\mathrm{inc}}$ is known (e.g., a plane wave), while the scattered field $\mathbf{E}^{\mathrm{sca}}$ arises from the induced surface current $\mathbf{J}$.

### 2.2 Tangential Field Components

Boundary conditions involve only the **tangential** component of the electric field. Define the tangential projection operator

```math
\mathbf{P}_t = \mathbf{I} - \hat{\mathbf{n}}\hat{\mathbf{n}}^T,
```

where $\hat{\mathbf{n}}$ is the unit normal vector to the surface. For any vector $\mathbf{v}$, the tangential component is

```math
\mathbf{v}_t = \mathbf{P}_t\mathbf{v} = \mathbf{v} - (\hat{\mathbf{n}}\cdot\mathbf{v})\hat{\mathbf{n}}.
```

Equivalently, using the cross-product notation often found in electromagnetics texts:

```math
\hat{\mathbf{n}} \times (\hat{\mathbf{n}} \times \mathbf{v}) = -\mathbf{v}_t.
```

The boundary conditions for PEC and impedance surfaces are expressed in terms of $\mathbf{E}_t^{\mathrm{tot}}$, the tangential total field.

### 2.3 Tangential Current Expansion

The unknown surface current $\mathbf{J}$ is purely tangential ($\hat{\mathbf{n}}\cdot\mathbf{J}=0$) and is expanded in a set of RWG basis functions $\{\mathbf{f}_n\}$:

```math
\mathbf{J}(\mathbf{r}) \approx \sum_{n=1}^N I_n \mathbf{f}_n(\mathbf{r}), \qquad \mathbf{r} \in \Gamma.
```

RWG functions are defined on pairs of triangles and ensure current continuity across edges. Their detailed properties are discussed in Chapter 3.

### 2.4 Physical Interpretation

- **Incident field**: The impressed source field that would exist in the absence of the scatterer.
- **Scattered field**: The field radiated by the induced surface currents, representing the scatterer's reaction.
- **Tangential projection**: Only the component parallel to the surface matters because the normal component of the electric field is discontinuous across a surface charge layer, while the tangential component is continuous (for finite surface impedance).

---

## 3. EFIE Operator and Mixed-Potential Form

### 3.1 Derivation from Maxwell's Equations

The Electric Field Integral Equation (EFIE) can be derived systematically from Maxwell's equations under the time-harmonic convention $e^{+i\omega t}$. Starting with Maxwell's equations in free space with sources:

```math
\begin{aligned}
\nabla \times \mathbf{E} &= -i\omega\mu_0 \mathbf{H}, \\
\nabla \times \mathbf{H} &= \mathbf{J} + i\omega\epsilon_0 \mathbf{E}, \\
\nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0}, \\
\nabla \cdot \mathbf{H} &= 0.
\end{aligned}
```

Introduce the magnetic vector potential $\mathbf{A}$ and electric scalar potential $\phi$ via the Lorenz gauge conditions:

```math
\mathbf{E} = -i\omega\mathbf{A} - \nabla\phi, \quad \mathbf{H} = \frac{1}{\mu_0}\nabla \times \mathbf{A}, \quad \nabla\cdot\mathbf{A} = -i\omega\mu_0\epsilon_0\phi.
```

The potentials satisfy the inhomogeneous Helmholtz equations:

```math
(\nabla^2 + k^2)\mathbf{A} = -\mu_0\mathbf{J}, \qquad (\nabla^2 + k^2)\phi = -\frac{\rho}{\epsilon_0},
```

where $k = \omega\sqrt{\mu_0\epsilon_0}$. Using the free-space Green's function $G(\mathbf{r},\mathbf{r}') = e^{-ikR}/(4\pi R)$ with $R = |\mathbf{r}-\mathbf{r}'|$, the solutions are

```math
\mathbf{A}(\mathbf{r}) = \mu_0 \int_{\Gamma} G(\mathbf{r},\mathbf{r}')\mathbf{J}(\mathbf{r}')\,dS', \qquad
\phi(\mathbf{r}) = \frac{1}{i\omega\epsilon_0} \int_{\Gamma} G(\mathbf{r},\mathbf{r}')\nabla'\cdot\mathbf{J}(\mathbf{r}')\,dS'.
```

The last expression uses the continuity equation $\nabla'\cdot\mathbf{J} = -i\omega\rho$. Substituting into $\mathbf{E}^{\mathrm{sca}} = -i\omega\mathbf{A} - \nabla\phi$ gives

```math
\mathbf{E}^{\mathrm{sca}}(\mathbf{r}) = -i\omega\mu_0 \int_{\Gamma} G(\mathbf{r},\mathbf{r}')\mathbf{J}(\mathbf{r}')\,dS'
- \nabla \frac{1}{i\omega\epsilon_0} \int_{\Gamma} G(\mathbf{r},\mathbf{r}')\nabla'\cdot\mathbf{J}(\mathbf{r}')\,dS'.
```

Applying the gradient operator under the integral and using $\nabla G = -\nabla' G$ (for functions of $\mathbf{r}-\mathbf{r}'$) leads to

```math
\mathbf{E}^{\mathrm{sca}}(\mathbf{r}) = -i\omega\mu_0 \int_{\Gamma} \left[\mathbf{I} + \frac{1}{k^2}\nabla\nabla\right] G(\mathbf{r},\mathbf{r}')\,\mathbf{J}(\mathbf{r}')\,dS'.
```

This is the **mixed-potential** representation of the scattered electric field.

### 3.2 Integral Operator for Scattered Field

The scattered electric field radiated by a surface current $\mathbf{J}$ is given by the integral operator

```math
\mathbf{E}^{\mathrm{sca}}(\mathbf{r}) = -i\omega\mu_0 \int_{\Gamma} \left[\mathbf{I} + \frac{1}{k^2}\nabla\nabla\right] G(\mathbf{r},\mathbf{r}')\,\mathbf{J}(\mathbf{r}')\,dS'.
```

This expression follows directly from the vector potential formulation $\mathbf{E}^{\mathrm{sca}} = -i\omega\mathbf{A} - \nabla\phi$ with

```math
\mathbf{A}(\mathbf{r}) = \mu_0 \int_{\Gamma} G(\mathbf{r},\mathbf{r}')\mathbf{J}(\mathbf{r}')\,dS', \qquad
\phi(\mathbf{r}) = \frac{1}{i\omega\epsilon_0} \int_{\Gamma} G(\mathbf{r},\mathbf{r}')\nabla'\cdot\mathbf{J}(\mathbf{r}')\,dS'.
```

The operator $\left[\mathbf{I} + \frac{1}{k^2}\nabla\nabla\right]$ couples the vector and scalar potentials; its action on the Green's function produces both the "identity" term (vector potential) and the "double-gradient" term (scalar potential).

### 3.3 Galerkin Testing and Integration by Parts

To discretize the EFIE using the Method of Moments with RWG basis functions $\{\mathbf{f}_n\}$, we employ Galerkin testing: multiply both sides by a test function $\mathbf{f}_m(\mathbf{r})$ and integrate over the surface $\Gamma$. The resulting matrix element is

```math
Z_{mn} = \langle \mathbf{f}_m, \mathcal{T}[\mathbf{f}_n] \rangle = -i\omega\mu_0 \iint_{\Gamma\times\Gamma} \mathbf{f}_m(\mathbf{r})\cdot\left[\mathbf{I} + \frac{1}{k^2}\nabla\nabla\right] G(\mathbf{r},\mathbf{r}')\,\mathbf{f}_n(\mathbf{r}')\,dS\,dS'.
```

The key step is **integration by parts** on the surface, which transfers one of the gradients from the Green's function onto the basis function. This transformation relies on the **surface divergence theorem**, which for smooth tangential vector fields $\mathbf{u}$ and scalar function $f$ on a surface $\Gamma$ states:

```math
\int_\Gamma (\nabla_s\cdot\mathbf{u})\, f\, dS = -\int_\Gamma \mathbf{u}\cdot\nabla_s f\, dS + \oint_{\partial\Gamma} (\mathbf{u}\cdot\hat{\mathbf{t}})\, f\, dl,
```

where $\nabla_s$ is the surface gradient, $\nabla_s\cdot$ is the surface divergence, $\hat{\mathbf{t}}$ is the unit tangent vector to the boundary $\partial\Gamma$, and $dl$ is the line element along the boundary.

For RWG basis functions defined on a closed surface or an open surface with no boundary (when considering interior edges only), the boundary term vanishes because:
1. $\mathbf{f}_m$ has zero normal component at the domain boundary (by construction for RWG functions on open surfaces, or because the surface is closed).
2. For interior edges, the support of $\mathbf{f}_m$ does not extend to the domain boundary.

#### Step-by-Step Integration by Parts

Let's perform the integration by parts in detail. Starting with the double-gradient term:

```math
I = \iint_{\Gamma\times\Gamma} \mathbf{f}_m(\mathbf{r})\cdot\nabla\nabla G(\mathbf{r},\mathbf{r}')\,\mathbf{f}_n(\mathbf{r}')\,dS\,dS'.
```

We treat $\nabla$ as acting on $\mathbf{r}$ and $\nabla'$ as acting on $\mathbf{r}'$. Note that $\nabla G(\mathbf{r},\mathbf{r}') = -\nabla' G(\mathbf{r},\mathbf{r}')$ because $G$ depends on $\mathbf{r}-\mathbf{r}'$.

**First integration by parts** (with respect to $\mathbf{r}$):
Apply the surface divergence theorem to the inner integral over $\mathbf{r}$:

```math
\int_\Gamma \mathbf{f}_m(\mathbf{r})\cdot\nabla\nabla G(\mathbf{r},\mathbf{r}')\,\mathbf{f}_n(\mathbf{r}')\,dS
= -\int_\Gamma \big[\nabla_s\cdot\mathbf{f}_m(\mathbf{r})\big]\,\nabla G(\mathbf{r},\mathbf{r}')\cdot\mathbf{f}_n(\mathbf{r}')\,dS + \text{boundary term}.
```

The boundary term vanishes for RWG functions as explained above. Thus:

```math
I = -\iint_{\Gamma\times\Gamma} \big[\nabla_s\cdot\mathbf{f}_m(\mathbf{r})\big]\,\nabla G(\mathbf{r},\mathbf{r}')\cdot\mathbf{f}_n(\mathbf{r}')\,dS\,dS'.
```

**Second integration by parts** (with respect to $\mathbf{r}'$):
Now apply the surface divergence theorem to the integral over $\mathbf{r}'$:

```math
\int_\Gamma \nabla G(\mathbf{r},\mathbf{r}')\cdot\mathbf{f}_n(\mathbf{r}')\,dS'
= -\int_\Gamma G(\mathbf{r},\mathbf{r}')\,\nabla_s'\cdot\mathbf{f}_n(\mathbf{r}')\,dS' + \text{boundary term}.
```

Again the boundary term vanishes. Substituting and using $\nabla G = -\nabla' G$ gives:

```math
\begin{aligned}
I &= -\iint_{\Gamma\times\Gamma} \big[\nabla_s\cdot\mathbf{f}_m(\mathbf{r})\big]\,\big[-\int_\Gamma G(\mathbf{r},\mathbf{r}')\,\nabla_s'\cdot\mathbf{f}_n(\mathbf{r}')\,dS'\big]\,dS \\
&= +\iint_{\Gamma\times\Gamma} \big[\nabla_s\cdot\mathbf{f}_m(\mathbf{r})\big]\,G(\mathbf{r},\mathbf{r}')\,\big[\nabla_s'\cdot\mathbf{f}_n(\mathbf{r}')\big]\,dS\,dS'.
\end{aligned}
```

Thus we obtain the final result:

```math
\iint_{\Gamma\times\Gamma} \mathbf{f}_m(\mathbf{r})\cdot\nabla\nabla G(\mathbf{r},\mathbf{r}')\,\mathbf{f}_n(\mathbf{r}')\,dS\,dS'
= -\iint_{\Gamma\times\Gamma} \big[\nabla_s\cdot\mathbf{f}_m(\mathbf{r})\big]\,G(\mathbf{r},\mathbf{r}')\,\big[\nabla_s'\cdot\mathbf{f}_n(\mathbf{r}')\big]\,dS\,dS',
```

where the overall minus sign comes from the fact that we started with $\nabla\nabla G$ and ended with a positive expression, but the original EFIE operator has a minus sign in front of the scalar potential term (see Equation $\eqref{eq:Esca}$).


### 3.4 Mixed-Potential Matrix Element

After integration by parts, we obtain the **mixed-potential form** of the EFIE matrix element:

```math
Z_{mn}^{\mathrm{EFIE}} = -i\omega\mu_0 \left[ \iint \mathbf{f}_m(\mathbf{r})\cdot\mathbf{f}_n(\mathbf{r}')\,G\,dS\,dS' - \frac{1}{k^2} \iint (\nabla_s\cdot\mathbf{f}_m(\mathbf{r}))(\nabla_s'\cdot\mathbf{f}_n(\mathbf{r}'))\,G\,dS\,dS' \right].
```

This is the exact expression computed by `assemble_Z_efie` in `src/assembly/EFIE.jl`. The two terms have clear physical interpretations:

- **Vector part** (first integral): Coupling through the magnetic vector potential $\mathbf{A}$; represents direct current‑current interaction.
- **Scalar part** (second integral): Coupling through the electric scalar potential $\phi$; represents charge‑charge interaction mediated by the surface divergence of the basis functions.

### 3.5 Physical Interpretation

The mixed‑potential split mirrors the decomposition of the electromagnetic field into its vector and scalar potentials:

```math
\mathbf{E}^{\mathrm{sca}} = -i\omega\mathbf{A} - \nabla\phi.
```

- The **vector part** corresponds to $-i\omega\mathbf{A}$: the electric field induced by the time‑varying magnetic vector potential.
- The **scalar part** corresponds to $-\nabla\phi$: the electric field due to gradients of the electric scalar potential, which is itself tied to surface charge via $\rho = -\frac{1}{i\omega}\nabla_s\cdot\mathbf{J}$.

For RWG basis functions, the surface divergence $\nabla_s\cdot\mathbf{f}_n$ is piecewise constant on each triangle, which simplifies the evaluation of the scalar part.

### 3.6 ASCII Diagram: EFIE Operator Structure

```
    Complete EFIE operator: E^sca(r) = T[J](r)
    
    T[J] = -iωμ₀ ∫ [I + (1/k²)∇∇] G(r,r') J(r') dS'
    
    After Galerkin testing with RWG basis:
    
    ┌─────────────────────────────────────────────────────────┐
    │              Matrix element Z_{mn}                      │
    │  = -iωμ₀ [ VECTOR_PART - SCALAR_PART ]                  │
    │                                                         │
    │  VECTOR_PART = ∫∫ f_m(r)·f_n(r') G(r,r') dS dS'         │
    │               (Current-current coupling)                │
    │                                                         │
    │  SCALAR_PART = (1/k²) ∫∫ [∇·f_m(r)] [∇'·f_n(r')]        │
    │               × G(r,r') dS dS'                          │
    │               (Charge-charge coupling)                  │
    └─────────────────────────────────────────────────────────┘
    
    Physical interpretation:
    
    Vector part ↔ Magnetic vector potential (A field)
    Scalar part ↔ Electric scalar potential (φ field)
    
    Implementation in assemble_Z_efie:
    Z[m,n] = -1im * omega_mu0 * (vec_part - scl_part)
```

### 3.7 Matrix Assembly Visualization

```
    For basis pair (m,n):
    
    Test basis f_m            Source basis f_n
         ▲                         ▲
         │                         │
         │                         │
    ┌────┼────┐               ┌────┼────┐
    │    │    │               │    │    │
    │ T+ │ T- │               │ T+ │ T- │
    │    │    │               │    │    │
    └────┼────┘               └────┼────┘
         │                         │
         │                         │
         ▼                         ▼
    
    Z_{mn} = sum over triangle pairs:
    - For each test triangle (T+_m or T-_m)
    - For each source triangle (T+_n or T-_n)
    - Compute quadrature over both triangles
    - Accumulate vector and scalar parts
```

**Implementation note:** Each RWG function is defined on a pair of triangles (T⁺ and T⁻). The matrix element $Z_{mn}$ therefore involves four triangle‑pair integrals: (T⁺_m, T⁺_n), (T⁺_m, T⁻_n), (T⁻_m, T⁺_n), and (T⁻_m, T⁻_n). The contributions are summed with appropriate signs according to the RWG definition.

---

## 4. Perfect Electric Conductor (PEC) Boundary Condition

### 4.1 Tangential Field Condition

For a perfect electric conductor, the tangential component of the total electric field must vanish on the surface:

```math
\mathbf{E}_t^{\mathrm{tot}}(\mathbf{r}) = \mathbf{0}, \qquad \mathbf{r} \in \Gamma.
```

This condition expresses the fact that a perfect conductor cannot sustain any tangential electric field; any incident field induces surface currents that exactly cancel the tangential field inside the conductor (and, by continuity, on its surface).

### 4.2 Relation between Scattered and Incident Fields

Substituting the field decomposition $\mathbf{E}_t^{\mathrm{tot}} = \mathbf{E}_t^{\mathrm{inc}} + \mathbf{E}_t^{\mathrm{sca}}$ yields

```math
\mathbf{E}_t^{\mathrm{inc}}(\mathbf{r}) + \mathbf{E}_t^{\mathrm{sca}}(\mathbf{r}) = \mathbf{0} \quad\Longrightarrow\quad \mathbf{E}_t^{\mathrm{sca}}(\mathbf{r}) = -\mathbf{E}_t^{\mathrm{inc}}(\mathbf{r}), \qquad \mathbf{r} \in \Gamma.
```

Thus, the scattered field must exactly oppose the incident field tangentially on the surface. This is the fundamental equation that determines the unknown surface current $\mathbf{J}$.

### 4.3 Galerkin Discretization

Insert the EFIE operator for $\mathbf{E}_t^{\mathrm{sca}}$ and expand $\mathbf{J}$ in RWG basis functions:

```math
\mathcal{T}[\mathbf{J}](\mathbf{r}) = -i\omega\mu_0 \int_{\Gamma} \left[\mathbf{I} + \frac{1}{k^2}\nabla\nabla\right] G(\mathbf{r},\mathbf{r}')\mathbf{J}(\mathbf{r}')\,dS' = -\mathbf{E}_t^{\mathrm{inc}}(\mathbf{r}).
```

Applying Galerkin testing with $\mathbf{f}_m$ gives the linear system

```math
\mathbf{Z}_{\mathrm{EFIE}} \mathbf{I} = \mathbf{V},
```

where

```math
Z_{mn} = \langle \mathbf{f}_m, \mathcal{T}[\mathbf{f}_n] \rangle, \qquad
V_m = -\langle \mathbf{f}_m, \mathbf{E}_t^{\mathrm{inc}} \rangle.
```

The matrix $\mathbf{Z}_{\mathrm{EFIE}}$ is precisely the mixed‑potential matrix derived in Section 3, and the right‑hand side vector $\mathbf{V}$ is the projection of the negative incident field onto the test functions.

### 4.4 Physical Interpretation

- **PEC condition**: No tangential electric field can exist at the surface because charges rearrange instantaneously to cancel it.
- **Induced currents**: The surface current $\mathbf{J}$ is the source of the scattered field that cancels the incident field.
- **Uniqueness**: For closed PEC surfaces at frequencies not coinciding with interior resonances, the EFIE has a unique solution. (Open surfaces require careful treatment of the zero‑frequency mode, discussed in Chapter 5.)

### 4.5 Three‑Line Algebraic Check

A quick derivation confirms the logic:

```math
\begin{aligned}
&\mathbf{E}_t^{\mathrm{tot}} = \mathbf{0} &&\text{(PEC condition)} \\
&\Rightarrow \mathbf{E}_t^{\mathrm{inc}} + \mathbf{E}_t^{\mathrm{sca}} = \mathbf{0} &&\text{(field decomposition)} \\
&\Rightarrow \mathbf{E}_t^{\mathrm{sca}} = -\mathbf{E}_t^{\mathrm{inc}} &&\text{(solve for scattered field)}
\end{aligned}
```

Testing both sides with $\mathbf{f}_m$ and using the linearity of $\mathcal{T}$ yields $\mathbf{Z}\mathbf{I} = \mathbf{V}$.

### 4.6 Example: Plane‑Wave Excitation

For a plane wave $\mathbf{E}^{\mathrm{inc}}(\mathbf{r}) = \mathbf{E}_0 e^{-i\mathbf{k}\cdot\mathbf{r}}$ (consistent with the $e^{+i\omega t}$ convention), the right‑hand side elements become

```math
V_m = -\int_{\Gamma} \mathbf{f}_m(\mathbf{r})\cdot\mathbf{E}_0 e^{-i\mathbf{k}\cdot\mathbf{r}}\,dS.
```

This integral is evaluated numerically using quadrature rules on each triangle of the RWG function's support. The implementation is found in `src/assembly/Excitation.jl`.

---

## 5. Impedance‑Sheet Boundary Condition

### 5.1 Generalized Boundary Condition

Many practical surfaces are not perfect conductors but exhibit a finite surface impedance $Z_s$ (units: ohms per square). The **impedance boundary condition** (IBC) states that the tangential total electric field is proportional to the surface current density:

```math
\mathbf{E}_t^{\mathrm{tot}}(\mathbf{r}) = Z_s(\mathbf{r})\,\mathbf{J}(\mathbf{r}), \qquad \mathbf{r} \in \Gamma.
```

This is a **local** approximation valid when the fields inside the material vary little over a skin depth and the surface can be treated as a sheet with effective impedance $Z_s$. The IBC reduces to the PEC condition when $Z_s = 0$ and to the perfect magnetic conductor (PMC) condition when $Z_s \to \infty$ (with appropriate scaling).

### 5.2 Derivation of the Modified EFIE

Insert the field decomposition $\mathbf{E}_t^{\mathrm{tot}} = \mathbf{E}_t^{\mathrm{inc}} + \mathbf{E}_t^{\mathrm{sca}}$:

```math
\mathbf{E}_t^{\mathrm{inc}}(\mathbf{r}) + \mathbf{E}_t^{\mathrm{sca}}(\mathbf{r}) = Z_s(\mathbf{r})\mathbf{J}(\mathbf{r}).
```

Rearranging gives

```math
\mathbf{E}_t^{\mathrm{sca}}(\mathbf{r}) - Z_s(\mathbf{r})\mathbf{J}(\mathbf{r}) = -\mathbf{E}_t^{\mathrm{inc}}(\mathbf{r}).
```

The term $-Z_s\mathbf{J}$ appears on the left‑hand side because the impedance contribution is subtracted from the scattered field.

### 5.3 Galerkin Discretization

Expanding $\mathbf{J} = \sum_n I_n \mathbf{f}_n$ and testing with $\mathbf{f}_m$ yields the linear system

```math
\bigl(\mathbf{Z}_{\mathrm{EFIE}} + \mathbf{Z}_{\mathrm{imp}}\bigr)\mathbf{I} = \mathbf{V},
```

where $\mathbf{Z}_{\mathrm{EFIE}}$ is the same PEC operator as before, $\mathbf{V}_m = -\langle \mathbf{f}_m, \mathbf{E}_t^{\mathrm{inc}} \rangle$, and the impedance matrix has entries

```math
(\mathbf{Z}_{\mathrm{imp}})_{mn} = -\langle \mathbf{f}_m, Z_s \mathbf{f}_n \rangle = -\int_{\Gamma} Z_s(\mathbf{r})\, \mathbf{f}_m(\mathbf{r})\cdot\mathbf{f}_n(\mathbf{r})\,dS.
```

The minus sign originates from the $-Z_s\mathbf{J}$ term in the rearranged boundary condition.

### 5.4 Patch‑Based Parameterization

In `DiffMoM.jl`, the surface impedance is approximated as **piecewise constant** on a set of $P$ patches that cover the surface. Let $\theta_p$ be the impedance value on patch $p$ (which may be real for resistive sheets or imaginary for reactive sheets). Then

```math
\mathbf{Z}_{\mathrm{imp}} = -\sum_{p=1}^{P} \theta_p \mathbf{M}_p,
```

where $\mathbf{M}_p$ is the **patch mass matrix** with entries

```math
(\mathbf{M}_p)_{mn} = \int_{\Gamma_p} \mathbf{f}_m(\mathbf{r})\cdot\mathbf{f}_n(\mathbf{r})\,dS,
```

and the integral is restricted to the portion of the surface belonging to patch $p$. The mass matrices are precomputed and stored for efficient assembly and differentiation.

### 5.5 ASCII Diagram: Impedance Boundary Condition

```
    Impedance surface: E_t^tot = Z_s J
    
    Field decomposition:
    
    E_t^tot = E_t^inc + E_t^sca = Z_s J
    
    Rearranged:
    
    E_t^sca - Z_s J = -E_t^inc
    
    After Galerkin testing:
    
    ┌─────────────────────────────────────────────────────────┐
    │         Complete linear system                          │
    │                                                         │
    │   [Z_EFIE + Z_imp] I = v                                │
    │                                                         │
    │   where:                                                │
    │   - Z_EFIE = standard EFIE operator                     │
    │   - Z_imp = impedance matrix                            │
    │   - v = -⟨f_m, E_t^inc⟩ (excitation)                   │
    │                                                         │
    │   Impedance matrix for patch p:                         │
    │   (Z_imp)_mn = -θ_p ⟨f_m, f_n⟩ over patch p            │
    │            = -θ_p (M_p)_mn                              │
    └─────────────────────────────────────────────────────────┘

    Patch-based parameterization:
    
    Surface divided into P patches:
    
    ┌─────┬─────┬─────┐
    │ θ₁  │ θ₂  │ θ₃  │
    ├─────┼─────┼─────┤
    │ θ₄  │ θ₅  │ θ₆  │
    ├─────┼─────┼─────┤
    │ θ₇  │ θ₈  │ θ₉  │
    └─────┴─────┴─────┘
    
    Each patch has uniform impedance θ_p
    Mass matrix M_p precomputed for each patch
```

### 5.6 Why the Impedance Term Carries a Minus Sign

The sign is a direct consequence of the algebra:

```math
\begin{aligned}
\mathbf{E}_t^{\mathrm{inc}} + \mathbf{E}_t^{\mathrm{sca}} &= Z_s\mathbf{J} \\
\Rightarrow \mathbf{E}_t^{\mathrm{sca}} - Z_s\mathbf{J} &= -\mathbf{E}_t^{\mathrm{inc}}.
\end{aligned}
```

The term $-Z_s\mathbf{J}$ appears on the left side; after testing with $\mathbf{f}_m$, it contributes $- \langle \mathbf{f}_m, Z_s\mathbf{f}_n \rangle$ to the matrix. Hence the impedance matrix is **subtracted** from the EFIE matrix (or, equivalently, added with a negative weight).

### 5.7 Physical Interpretation

- **Resistive sheet** ($Z_s = R_s$ real): Dissipates power; the scattered field is reduced compared to a PEC, and part of the incident energy is absorbed.
- **Reactive sheet** ($Z_s = iX_s$ imaginary): Stores energy reactively; the surface behaves inductively ($X_s > 0$) or capacitively ($X_s < 0$), shifting resonance frequencies.
- **Patch approximation**: Allows spatially varying impedance profiles, useful for modeling graded metasurfaces or optimizing impedance distributions.

The implementation of impedance matrices and their derivatives is found in `src/assembly/Impedance.jl`, and the full system assembly is performed by `assemble_full_Z` in `src/solver/Solve.jl`.

---

## 6. Implementation Details and Sign Conventions

### 6.1 Sign Consistency in the Code

The Julia implementation follows the mathematical derivation exactly. In `src/assembly/EFIE.jl`, the matrix assembly uses

```julia
omega_mu0 = k * eta0
Z[m, n] = -1im * omega_mu0 * val
```

where `val` is the real‑valued result of the double integral (vector part minus scalar part). The factor `-1im` corresponds to $-i$ in the formula

```math
Z_{mn}^{\mathrm{EFIE}} = -i\omega\mu_0 \bigl(\text{vector part} - \text{scalar part}\bigr).
```

Since $\omega\mu_0 = k\eta_0$ (with $\eta_0 = \sqrt{\mu_0/\epsilon_0}$), the code computes `omega_mu0 = k * eta0` for efficiency.

### 6.2 Cross‑Validation with Other Conventions

When comparing with codes or textbooks that use the $e^{-i\omega t}$ convention, remember:

1. The sign of the imaginary unit in front of $\omega\mu_0$ flips (becomes $+i$).
2. The exponential in the Green's function changes from $e^{-ikR}$ to $e^{+ikR}$.
3. The overall sign of the matrix may differ by a factor of $-1$ if the impedance term is defined with opposite sign.

Always verify the time convention first; `DiffMoM.jl` consistently uses $e^{+i\omega t}$ throughout.

### 6.3 Verification by Consistency Checks

A simple way to verify your understanding is to run the following sanity checks:

1. **PEC limit**: Set all impedance parameters to zero; `assemble_full_Z` should return the same matrix as `assemble_Z_efie` (within rounding error).
2. **Reciprocity**: For PEC and reciprocal impedance sheets, the EFIE matrix should be symmetric (up to numerical integration errors).
3. **Energy conservation**: For a lossless PEC scatterer, the power radiated (computed via the far‑field) should equal the power extracted from the incident field.

These checks are implemented in the test suite of the package.

## 7. Practical Examples

### 7.1 Minimal Forward Assembly

The following snippet constructs the EFIE matrix for a PEC plate at 3 GHz:

```julia
using DiffMoM

freq = 3e9                     # 3 GHz
c0   = 299792458.0             # speed of light
k    = 2π / (c0 / freq)        # wavenumber

# Create a rectangular plate discretized with 6×6 quads (12×12 triangles)
mesh = make_rect_plate(0.1, 0.1, 6, 6)
rwg  = build_rwg(mesh)         # RWG basis functions
Zef  = assemble_Z_efie(mesh, rwg, k; quad_order=3)
```

The resulting `Zef` is a complex dense matrix of size $N \times N$, where $N$ is the number of RWG edges.

### 7.2 Adding Impedance Sheets

To include an impedance sheet described by a patch parameter vector `theta` (length $P$), use:

```julia
theta = zeros(ComplexF64, P)   # PEC by default
# Set some patches to resistive or reactive values
theta[1] = 50.0                # 50 Ω resistive
theta[2] = 100.0im             # 100 Ω inductive

Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3)
Z_full = assemble_full_Z(Z_efie, Mp, theta)
```

The function `assemble_full_Z` adds the impedance contribution $-\sum_p \theta_p \mathbf{M}_p$ to the EFIE matrix.

### 7.3 Solving the Linear System

With the matrix assembled, solve for the surface current coefficients:

```julia
# Plane‑wave excitation
E0 = 1.0                        # amplitude (V/m)
k_vec = Vec3(0.0, 0.0, k)       # propagating in +z direction
pol = Vec3(1.0, 0.0, 0.0)       # x-polarization
pw = make_plane_wave(k_vec, E0, pol)
V = assemble_excitation(mesh, rwg, pw)   # right‑hand side

# Solve (using LU factorization for small problems)
I = Z_full \ V
```

The vector `I` contains the complex expansion coefficients for the RWG basis functions.

### 7.4 Consistency Check

A useful debugging step: when `theta = zeros(...)`, the matrix `Z_full` should match `Zef` to within machine precision. This confirms that the impedance term is correctly added with the proper sign.

## 8. Code Mapping

- **Green's function**: `src/basis/Greens.jl` – implements $G(\mathbf{r},\mathbf{r}') = e^{-ikR}/(4\pi R)$.
- **EFIE matrix assembly**: `src/assembly/EFIE.jl` – computes the mixed‑potential matrix elements `Z[m,n]`.
- **Excitation vector**: `src/assembly/Excitation.jl` – builds the right‑hand side vector $\mathbf{V}$ for plane‑wave or dipole sources.
- **Impedance matrices**: `src/assembly/Impedance.jl` – precomputes patch mass matrices $\mathbf{M}_p$ and their derivatives.
- **Full system assembly**: `src/solver/Solve.jl` – function `assemble_full_Z` combines EFIE and impedance matrices.

## 9. Exercises and Problems

### 9.1 Conceptual Questions

1. **Sign convention**: Starting from Maxwell's equations with the $e^{+i\omega t}$ convention, show that the wave equation for the electric field becomes $(\nabla^2 + k^2)\mathbf{E} = \mathbf{0}$ in source‑free regions. How does the sign change if $e^{-i\omega t}$ is used?

2. **Mixed‑potential derivation**: Perform the integration‑by‑parts step that transforms
   $\iint \mathbf{f}_m\cdot\nabla\nabla G\,\mathbf{f}_n\,dS\,dS'$
   into
   $-\iint (\nabla_s\cdot\mathbf{f}_m) G (\nabla_s'\cdot\mathbf{f}_n)\,dS\,dS'$.
   State the surface‑divergence theorem used and explain why boundary terms vanish for RWG functions.

3. **Impedance sign**: Derive the impedance matrix entry $(\mathbf{Z}_{\mathrm{imp}})_{mn} = -\langle \mathbf{f}_m, Z_s\mathbf{f}_n\rangle$ directly from the boundary condition $\mathbf{E}_t^{\mathrm{tot}} = Z_s\mathbf{J}$. Why does the minus sign appear, and what would happen if it were omitted?

### 9.2 Numerical Exercises

```julia
using DiffMoM

# Exercise 1: Verify PEC‑impedance consistency
function check_pec_limit()
    mesh = make_rect_plate(0.05, 0.05, 4, 4)
    rwg = build_rwg(mesh)
    k = 2π / 0.1  # λ = 0.1 m
    Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3)

    partition = assign_patches_grid(mesh; nx=2, ny=2, nz=1)
    P = partition.P  # number of patches
    theta = zeros(ComplexF64, P)
    Mp = precompute_patch_mass(mesh, rwg, partition)
    Z_full = assemble_full_Z(Z_efie, Mp, theta)
    
    return norm(Z_full - Z_efie) / norm(Z_efie)
end

# Exercise 2: Symmetry check
function check_symmetry()
    mesh = read_obj_mesh("sphere.obj")   # sphere meshes should be created externally or loaded via read_obj_mesh
    rwg = build_rwg(mesh)
    k = 2π / 0.15
    Z = assemble_Z_efie(mesh, rwg, k; quad_order=3)
    return norm(Z - Z') / norm(Z)
end
```

### 9.3 Derivation Problems

1. **PEC as a limiting case**: Show that the impedance boundary condition $\mathbf{E}_t^{\mathrm{tot}} = Z_s\mathbf{J}$ reduces to the PEC condition $\mathbf{E}_t^{\mathrm{tot}} = \mathbf{0}$ when $Z_s \to 0$, provided the current remains finite. What happens if $Z_s \to \infty$?

2. **Energy interpretation**: For a resistive sheet with $Z_s = R_s > 0$, the power absorbed per unit area is $\frac{1}{2}R_s|\mathbf{J}|^2$ (time‑average). Starting from the Poynting vector, show how this absorption corresponds to a dissipative (positive-real) contribution in the system operator under the chosen sign convention.

## 10. Summary and Key Insights

### 10.1 The Mixed‑Potential EFIE

- The EFIE can be written in a **mixed‑potential form** that separates vector (current‑current) and scalar (charge‑charge) couplings.
- Integration by parts transfers the double‑gradient operator onto the basis function divergences, yielding a numerically stable representation.
- This form is implemented directly in `assemble_Z_efie` and forms the core of the MoM solver.

### 10.2 Boundary Conditions

- **PEC**: $\mathbf{E}_t^{\mathrm{tot}} = \mathbf{0}$ leads to $\mathbf{E}_t^{\mathrm{sca}} = -\mathbf{E}_t^{\mathrm{inc}}$, producing the standard EFIE system $\mathbf{Z}\mathbf{I} = \mathbf{V}$.
- **Impedance sheet**: $\mathbf{E}_t^{\mathrm{tot}} = Z_s\mathbf{J}$ adds an extra term $-\langle \mathbf{f}_m, Z_s\mathbf{f}_n\rangle$ to the matrix, resulting in $(\mathbf{Z}_{\mathrm{EFIE}} + \mathbf{Z}_{\mathrm{imp}})\mathbf{I} = \mathbf{V}$.
- The **minus sign** in the impedance term arises from moving $Z_s\mathbf{J}$ to the left‑hand side of the equation.

### 10.3 Implementation Notes

- The package uses the $e^{+i\omega t}$ convention; the Green's function is $e^{-ikR}/(4\pi R)$.
- Impedance values can be specified per patch, enabling spatially varying surface properties.
- Consistency checks (PEC limit, symmetry, energy conservation) are essential for verifying correctness.

## 11. Chapter Checklist

Before proceeding to Chapter 3, ensure you understand:

- [ ] The derivation of the mixed‑potential EFIE matrix element $Z_{mn}^{\mathrm{EFIE}}$.
- [ ] How integration by parts transfers the double‑gradient operator onto basis function divergences.
- [ ] The physical interpretation of the vector and scalar parts (magnetic vector potential vs. electric scalar potential).
- [ ] The PEC boundary condition and its translation to the linear system $\mathbf{Z}\mathbf{I} = \mathbf{V}$.
- [ ] The impedance boundary condition and why it adds a term $-\langle \mathbf{f}_m, Z_s\mathbf{f}_n\rangle$ to the matrix.
- [ ] How patch‑based impedance parameterization leads to $\mathbf{Z}_{\mathrm{imp}} = -\sum_p \theta_p \mathbf{M}_p$.
- [ ] The sign conventions used in the code and how to cross‑validate with other references.

If any items are unclear, review the relevant sections or consult the mathematical prerequisites appendix.

## 12. Further Reading

1. **Classical references on EFIE and boundary conditions**:
   - Harrington, R. F. (1993). *Field Computation by Moment Methods*, Chapters 3‑4. IEEE Press.
   - Chew, W. C. (1995). *Waves and Fields in Inhomogeneous Media*, Chapter 6. IEEE Press.
   - Volakis, J. L., & Sertel, K. (2012). *Integral Equation Methods for Electromagnetics*, Chapters 2‑3. SciTech Publishing.

2. **Impedance boundary conditions**:
   - Senior, T. B. A., & Volakis, J. L. (1995). *Approximate Boundary Conditions in Electromagnetics*. IET.
   - Tretyakov, S. A. (2003). *Analytical Modeling in Applied Electromagnetics*. Artech House.

3. **Numerical implementation details**:
   - Peterson, A. F., et al. (1998). *Computational Methods for Electromagnetics*, Chapter 8. IEEE Press.
   - Graglia, R. D., & Lombardi, G. (2004). *Singular Integrals in Integral Equations and Moment Methods*. IEEE Press.

---

*Next: In Chapter 3, we will examine the Rao‑Wilton‑Glisson (RWG) basis functions in detail, discussing their definition, properties, and role in enforcing current continuity and charge conservation.*
