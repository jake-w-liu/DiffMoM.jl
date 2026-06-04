# Why Integral Equations

## Purpose

This chapter establishes the mathematical and physical foundations of the Integral Equation Method of Moments (IE-MoM) approach. If you are familiar with Maxwell's equations but new to computational electromagnetics, the key conceptual shift is understanding how surface integral equations transform volumetric scattering problems into boundary-only formulations.

We will derive the Electric Field Integral Equation (EFIE) from first principles, explain why surface methods are particularly well-suited for open-region scattering, and compare the computational trade-offs with volume-based methods like Finite Element Method (FEM) and Finite Difference Time Domain (FDTD).

---

## Learning Goals

After this chapter, you should be able to:

1. Derive the EFIE from Maxwell's equations using Green's function theory.
2. Explain the physical meaning of surface equivalence principles.
3. Compare computational complexity of surface vs. volume discretization methods.
4. Understand the Method of Moments (MoM) framework for discretizing integral equations.
5. Estimate memory and computational requirements for MoM simulations.

---

## 1. Maxwell's Equations and the Scattering Problem

### 1.1 Time-Harmonic Maxwell's Equations

We begin with time-harmonic Maxwell's equations in free space (using the $e^{+i\omega t}$ convention):

```math
\begin{aligned}
\nabla \times \mathbf{E} &= -i\omega\mu_0 \mathbf{H} \\
\nabla \times \mathbf{H} &= i\omega\epsilon_0 \mathbf{E} + \mathbf{J} \\
\nabla \cdot \mathbf{E} &= \rho/\epsilon_0 \\
\nabla \cdot \mathbf{H} &= 0
\end{aligned}
```

For scattering problems, we decompose the total field into incident and scattered components:

```math
\begin{aligned}
\mathbf{E}^{\text{tot}}(\mathbf{r}) &= \mathbf{E}^{\text{inc}}(\mathbf{r}) + \mathbf{E}^{\text{sca}}(\mathbf{r}) \\
\mathbf{H}^{\text{tot}}(\mathbf{r}) &= \mathbf{H}^{\text{inc}}(\mathbf{r}) + \mathbf{H}^{\text{sca}}(\mathbf{r})
\end{aligned}
```

The incident field $(\mathbf{E}^{\text{inc}}, \mathbf{H}^{\text{inc}})$ is known (e.g., a plane wave), while the scattered field $(\mathbf{E}^{\text{sca}}, \mathbf{H}^{\text{sca}})$ arises from interaction with scattering objects.

### 1.2 Boundary Conditions for Perfect Electric Conductors (PEC)

For a perfect electric conductor (PEC), the tangential component of the total electric field must vanish on the surface $\Gamma$:

```math
\hat{\mathbf{n}} \times \mathbf{E}^{\text{tot}}(\mathbf{r}) = \mathbf{0}, \quad \mathbf{r} \in \Gamma
```

where $\hat{\mathbf{n}}$ is the outward unit normal to the surface. This implies:

```math
\hat{\mathbf{n}} \times \mathbf{E}^{\text{sca}}(\mathbf{r}) = -\hat{\mathbf{n}} \times \mathbf{E}^{\text{inc}}(\mathbf{r}), \quad \mathbf{r} \in \Gamma
```

The scattered field is radiated by equivalent surface currents $\mathbf{J}_s$ and $\mathbf{M}_s$ on $\Gamma$, but for PEC scattering, we only need the electric surface current $\mathbf{J}_s$ (since $\mathbf{M}_s = \mathbf{0}$ on PEC).

### 1.3 Impedance Boundary Condition (Generalization)

For surfaces with finite impedance $Z_s$, the boundary condition is:

```math
\hat{\mathbf{n}} \times \mathbf{E}^{\text{tot}}(\mathbf{r}) = Z_s \mathbf{J}_s(\mathbf{r}), \quad \mathbf{r} \in \Gamma
```

where $\mathbf{J}_s$ is the surface current density. This reduces to the PEC case when $Z_s = 0$.

---

## 2. Green's Function Solution of Maxwell's Equations

### 2.1 Free-Space Green's Function

The free-space scalar Green's function $G(\mathbf{r},\mathbf{r}')$ satisfies the inhomogeneous Helmholtz equation:

```math
(\nabla^2 + k^2) G(\mathbf{r},\mathbf{r}') = -\delta(\mathbf{r} - \mathbf{r}')
```

with outgoing wave boundary condition, where $k = \omega\sqrt{\mu_0\epsilon_0}$ is the wavenumber. The solution is:

```math
G(\mathbf{r},\mathbf{r}') = \frac{e^{-ik|\mathbf{r} - \mathbf{r}'|}}{4\pi|\mathbf{r} - \mathbf{r}'|}
```

This Green's function automatically satisfies the radiation condition at infinity, which is a key advantage of integral equation methods.

### 2.2 Vector Potential Formulation

The scattered electric field due to a surface current $\mathbf{J}_s$ can be expressed using the magnetic vector potential $\mathbf{A}$:

```math
\begin{aligned}
\mathbf{A}(\mathbf{r}) &= \mu_0 \int_\Gamma G(\mathbf{r},\mathbf{r}') \mathbf{J}_s(\mathbf{r}') \, dS' \\
\mathbf{E}^{\text{sca}}(\mathbf{r}) &= -i\omega\mathbf{A}(\mathbf{r}) - \nabla\phi(\mathbf{r}) \\
\phi(\mathbf{r}) &= \frac{1}{i\omega\epsilon_0} \int_\Gamma G(\mathbf{r},\mathbf{r}') \nabla' \cdot \mathbf{J}_s(\mathbf{r}') \, dS'
\end{aligned}
```

where $\phi$ is the electric scalar potential. Combining these gives the electric field integral operator:

```math
\mathbf{E}^{\text{sca}}(\mathbf{r}) = -i\omega\mu_0 \int_\Gamma \left[\mathbf{I} + \frac{1}{k^2}\nabla\nabla\right] G(\mathbf{r},\mathbf{r}') \mathbf{J}_s(\mathbf{r}') \, dS'
```

This is the core operator that maps surface currents to scattered fields.

---

## 3. Derivation of the Electric Field Integral Equation (EFIE)

### 3.1 Tangential Projection

Applying the tangential projection operator $\hat{\mathbf{n}} \times (\hat{\mathbf{n}} \times \cdot)$ to the scattered field expression and enforcing the PEC boundary condition gives:

```math
\hat{\mathbf{n}} \times \hat{\mathbf{n}} \times \mathbf{E}^{\text{sca}}(\mathbf{r}) = -\hat{\mathbf{n}} \times \hat{\mathbf{n}} \times \mathbf{E}^{\text{inc}}(\mathbf{r}), \quad \mathbf{r} \in \Gamma
```

Substituting the integral operator leads to the Electric Field Integral Equation (EFIE):

```math
\hat{\mathbf{n}} \times \hat{\mathbf{n}} \times \left[-i\omega\mu_0 \int_\Gamma \left(\mathbf{I} + \frac{1}{k^2}\nabla\nabla\right) G(\mathbf{r},\mathbf{r}') \mathbf{J}_s(\mathbf{r}') \, dS'\right] = -\hat{\mathbf{n}} \times \hat{\mathbf{n}} \times \mathbf{E}^{\text{inc}}(\mathbf{r})
```

### 3.2 Mixed-Potential Form

A more computationally friendly form is obtained by integrating the $\nabla\nabla$ operator by parts:

```math
\begin{aligned}
\mathbf{E}^{\text{sca}}(\mathbf{r}) = &-i\omega\mu_0 \int_\Gamma G(\mathbf{r},\mathbf{r}') \mathbf{J}_s(\mathbf{r}') \, dS' \\
&+ \frac{i}{\omega\epsilon_0} \nabla \int_\Gamma G(\mathbf{r},\mathbf{r}') \nabla' \cdot \mathbf{J}_s(\mathbf{r}') \, dS'
\end{aligned}
```

This separates the vector potential contribution (first term) from the scalar potential contribution (second term), which is crucial for stable numerical implementation.

### 3.3 Why This is an Integral Equation

The EFIE has the form:

```math
\mathcal{T}[\mathbf{J}_s](\mathbf{r}) = -\mathbf{E}^{\text{inc}}_t(\mathbf{r}), \quad \mathbf{r} \in \Gamma
```

where $\mathcal{T}$ is a linear integral operator mapping the unknown surface current $\mathbf{J}_s$ to the tangential scattered field. This is a **Fredholm integral equation of the first kind**—the unknown function appears only inside the integral.

---

## 4. Surface vs. Volume Discretization: Dimensional Analysis

### 4.1 Scaling Laws for Unknown Counts

Consider scattering from an object with characteristic size $L$ discretized with resolution $h$ (typical element size). The number of unknowns scales as:

- **Volume methods (FEM, FDTD):** $N_{\text{vol}} \sim (L/h)^3$
- **Surface methods (MoM, BEM):** $N_{\text{surf}} \sim (L/h)^2$

For example, with $L = 1\ \text{m}$ and $h = 1\ \text{cm}$:
- Volume: $N_{\text{vol}} \sim 100^3 = 10^6$ unknowns
- Surface: $N_{\text{surf}} \sim 100^2 = 10^4$ unknowns

The surface method reduces the unknown count by **one order of magnitude** in terms of $L/h$.

### 4.2 Why This Matters for High-Frequency Problems

At high frequencies, $h$ must be small relative to wavelength $\lambda$ (typically $h < \lambda/10$). As frequency increases:
- $\lambda$ decreases → $h$ must decrease
- $L/h$ increases rapidly
- Volume methods become prohibitively expensive due to cubic scaling

Surface methods mitigate this through quadratic rather than cubic scaling, making them attractive for high-frequency scattering.

### 4.3 The Dense Matrix Trade-off

While surface methods have fewer unknowns, the EFIE operator $\mathcal{T}$ is **non-local**—every basis function couples to every other basis function through the Green's function. This leads to dense matrices:

- **Storage:** $O(N^2)$ for dense $N \times N$ matrix
- **Direct solve cost:** $O(N^3)$ for LU factorization
- **Matrix-vector multiply:** $O(N^2)$ (no sparsity)

This is the fundamental trade-off: fewer unknowns but dense coupling versus more unknowns but sparse coupling (in volume methods).

---

## 5. The Method of Moments (MoM) Framework

### 5.1 Basis Expansion

The Method of Moments discretizes the integral equation by expanding the unknown surface current in a finite basis:

```math
\mathbf{J}_s(\mathbf{r}) \approx \sum_{n=1}^N I_n \mathbf{f}_n(\mathbf{r})
```

where $\{\mathbf{f}_n\}_{n=1}^N$ are vector basis functions defined on the surface $\Gamma$, and $\{I_n\}_{n=1}^N$ are complex expansion coefficients.

### 5.2 Testing (Weighted Residual Method)

We form the residual:

```math
\mathcal{R}(\mathbf{r}) = \mathcal{T}\left[\sum_{n=1}^N I_n \mathbf{f}_n\right](\mathbf{r}) + \mathbf{E}^{\text{inc}}_t(\mathbf{r})
```

and enforce orthogonality to a set of testing functions $\{\mathbf{w}_m\}_{m=1}^N$:

```math
\langle \mathbf{w}_m, \mathcal{R} \rangle = 0, \quad m = 1,\dots,N
```

where $\langle \cdot, \cdot \rangle$ denotes an appropriate inner product on $\Gamma$.

### 5.3 Galerkin Method

In the Galerkin approach, we choose testing functions equal to basis functions: $\mathbf{w}_m = \mathbf{f}_m$. This leads to a symmetric system (for self-adjoint operators) and often provides optimal convergence properties.

The discretized system becomes:

```math
\sum_{n=1}^N Z_{mn} I_n = V_m, \quad m = 1,\dots,N
```

with matrix elements:

```math
\begin{aligned}
Z_{mn} &= \langle \mathbf{f}_m, \mathcal{T}[\mathbf{f}_n] \rangle \\
V_m &= -\langle \mathbf{f}_m, \mathbf{E}^{\text{inc}}_t \rangle
\end{aligned}
```

### 5.4 Rao-Wilton-Glisson (RWG) Basis Functions

This package uses Rao-Wilton-Glisson (RWG) basis functions, which are defined on pairs of triangular elements sharing a common edge. RWG functions:
1. Ensure current continuity across element boundaries
2. Have linear variation within each triangle
3. Enforce charge conservation through divergence properties
4. Provide a natural physical interpretation (edge currents)

We will explore RWG functions in detail in Chapter 3.

---

## 6. Comparison with Other Numerical Methods

### 6.1 Finite Element Method (FEM)
- **Domain:** Discretizes volume surrounding scatterer
- **Matrix:** Sparse, banded structure
- **Boundary conditions:** Requires artificial absorbing boundary conditions (ABCs) or perfectly matched layers (PMLs)
- **Advantages:** Handles inhomogeneous media naturally
- **Disadvantages:** Larger unknown count for open-region problems

### 6.2 Finite Difference Time Domain (FDTD)
- **Domain:** Discretizes volume with staggered grid
- **Approach:** Solves Maxwell's equations directly in time domain
- **Boundary conditions:** Requires ABCs/PMLs
- **Advantages:** Broadband results from single simulation, intuitive
- **Disadvantages:** Staircasing errors, difficult for curved surfaces

### 6.3 Method of Moments (MoM)
- **Domain:** Discretizes only the scattering surface
- **Matrix:** Dense, full coupling
- **Boundary conditions:** Built into Green's function (radiation condition)
- **Advantages:** Exact radiation condition, fewer unknowns for smooth surfaces
- **Disadvantages:** Dense matrices limit problem size, difficult for inhomogeneous media

### 6.4 Hybrid Methods

In practice, hybrid methods combine strengths:
- **FEM-MoM:** FEM for complex interior regions, MoM for exterior radiation
- **MLFMM-MoM:** Multilevel fast multipole method accelerates MoM matrix-vector products
- **IE-FFT:** Integral equation with fast Fourier transform acceleration

This package focuses on the **dense MoM reference implementation** for accuracy verification and gradient-based optimization studies.

---

## 7. Computational Complexity Analysis

### 7.1 Memory Requirements

For $N$ RWG unknowns, a dense complex matrix requires:

```math
\text{Memory (GiB)} = \frac{16 N^2}{1024^3}
```

where 16 bytes = 8 bytes for real part + 8 bytes for imaginary part (double precision complex).

### 7.2 Operation Counts

- **Matrix fill:** $O(N^2)$ operations (each $Z_{mn}$ requires numerical integration)
- **Direct solve (LU):** $O(N^3)$ operations
- **Matrix-vector multiply:** $O(N^2)$ operations
- **Iterative solve (GMRES/Krylov):** $O(kN^2)$ operations for $k$ iterations

### 7.3 Practical Scaling Limits

For a typical workstation with 64 GB RAM:
- Maximum $N \approx \sqrt{64\times 1024^3 / 16} \approx 65,\!000$
- LU factorization time: $O((6.5\times 10^4)^3) \sim 2.7\times 10^{14}$ operations
- At 100 GFLOP/s: $\sim 2.7\times 10^3$ seconds $\approx 45$ minutes

This illustrates why dense MoM is practical for moderate-sized problems ($N < 10^4$) but requires acceleration techniques (FMM, FFT) or iterative methods for larger problems.

---

## 8. What This Package Solves

The `DiffMoM.jl` package solves the discretized EFIE system:

```math
\mathbf{Z}\mathbf{I} = \mathbf{V}
```

where:
- $\mathbf{Z} \in \mathbb{C}^{N\times N}$ is the MoM impedance matrix
- $\mathbf{I} \in \mathbb{C}^{N}$ is the vector of RWG expansion coefficients
- $\mathbf{V} \in \mathbb{C}^{N}$ is the excitation vector from incident field

Once $\mathbf{I}$ is computed, all derived quantities are obtained through linear or quadratic operations:

```math
\begin{aligned}
\text{Far field:} & \quad \mathbf{E}^\infty = \mathbf{G}\mathbf{I} \quad (\text{linear}) \\
\text{Objective:} & \quad J = \mathbf{I}^\dagger\mathbf{Q}\mathbf{I} \quad (\text{quadratic}) \\
\text{Gradients:} & \quad \frac{\partial J}{\partial\theta_p} \text{ via adjoint method}
\end{aligned}
```

This structure enables efficient gradient computation for optimization, which is the focus of Part III of this documentation.

---

## 9. Summary and Key Insights

### 9.1 Why Integral Equations for Open-Region Scattering?

1. **Exact radiation condition:** The free-space Green's function $G(\mathbf{r},\mathbf{r}')$ automatically satisfies the Sommerfeld radiation condition, eliminating the need for artificial absorbing boundaries.

2. **Dimensional reduction:** Surface discretization reduces the unknown count from $O((L/h)^3)$ to $O((L/h)^2)$, where $L$ is object size and $h$ is mesh resolution.

3. **Physical transparency:** Surface currents $\mathbf{J}_s$ have direct physical interpretation as actual induced currents on conductors.

4. **Accuracy for smooth surfaces:** For perfectly conducting or impedance surfaces, the surface current representation is mathematically exact (no volumetric field approximation needed).

### 9.2 Trade-offs and Practical Considerations

1. **Dense matrices:** While having fewer unknowns, MoM produces fully populated matrices requiring $O(N^2)$ storage and $O(N^3)$ direct solve time.

2. **Singular integrals:** The Green's function singularity at $\mathbf{r} = \mathbf{r}'$ requires special quadrature techniques (discussed in Chapter 4).

3. **Frequency scaling:** MoM complexity increases with frequency due to:
   - Finer mesh required ($h \propto \lambda$)
   - Larger electrical size ($L/\lambda$ increases)
   - Potential ill-conditioning at low frequencies

4. **Acceleration methods:** For large problems, fast algorithms (FMM, FFT, MLFMA) reduce matrix-vector products from $O(N^2)$ to $O(N\log N)$ or $O(N)$.

### 9.3 The DiffMoM.jl Philosophy

This package implements a **reference dense MoM solver** with emphasis on:

1. **Operator transparency:** Clear mapping between mathematical formulas and code implementation.
2. **Differentiable design:** Accurate gradient computation via adjoint method for optimization.
3. **Verification focus:** Internal consistency checks and validation against analytical solutions.
4. **Educational value:** Code structured to teach MoM fundamentals while providing research capabilities.

While dense MoM limits problem size, it provides a gold standard for verifying accelerated methods and serves as an ideal testbed for gradient-based optimization studies.

---

## 10. Exercises and Problems

### 10.1 Conceptual Questions

1. **Derivation practice:** Starting from Maxwell's equations in differential form, derive the vector potential expression for the scattered electric field:
   $\mathbf{E}^{\text{sca}} = -i\omega\mathbf{A} - \nabla\phi$.

2. **Boundary condition analysis:** Explain why the impedance boundary condition $\hat{\mathbf{n}} \times \mathbf{E}^{\text{tot}} = Z_s \mathbf{J}_s$ reduces to the PEC condition when $Z_s = 0$, and to the PMC (perfect magnetic conductor) condition when $Z_s \to \infty$.

3. **Green's function properties:** Prove that the free-space Green's function $G(\mathbf{r},\mathbf{r}') = e^{-ikR}/(4\pi R)$ satisfies:
   - The Helmholtz equation: $(\nabla^2 + k^2)G = -\delta(\mathbf{r}-\mathbf{r}')$
   - The reciprocity relation: $G(\mathbf{r},\mathbf{r}') = G(\mathbf{r}',\mathbf{r})$
   - The far-field approximation: $G(\mathbf{r},\mathbf{r}') \sim e^{-ikr}/(4\pi r) e^{ik\hat{\mathbf{r}}\cdot\mathbf{r}'}$ as $r \to \infty$

### 10.2 Numerical Estimation Problems

1. **Memory estimation:** For a spherical scatterer of radius $a = 1\ \text{m}$ at frequency $f = 300\ \text{MHz}$ ($\lambda = 1\ \text{m}$), estimate the number of RWG unknowns $N$ assuming 10 triangles per wavelength squared. Compute the dense matrix memory requirement in GiB.

2. **Scaling analysis:** Show analytically that doubling the electrical size of an object (keeping mesh density constant in terms of $\lambda$) increases:
   - Surface unknown count by 4×
   - Dense matrix memory by 16×  
   - Direct solve time by 64× (for $O(N^3)$ scaling)

3. **Comparison study:** Compare MoM and FEM for a $1\ \text{m} \times 1\ \text{m}$ plate at 3 GHz ($\lambda = 0.1\ \text{m}$). Assume:
   - MoM: 100 triangles/$\lambda^2$, $h = \lambda/10$
   - FEM: Volume extends $\lambda/2$ from plate, 10 cells/$\lambda$
   Compute and compare unknown counts for both methods.

### 10.3 Implementation Exercises

```julia
using DiffMoM

# Exercise 1: Mesh size estimation
function estimate_mesh_requirements(a::Float64, f::Float64, triangles_per_lambda2::Float64)
    c0 = 299792458.0
    λ = c0 / f
    surface_area = 4π * a^2  # Sphere
    N_triangles = triangles_per_lambda2 * surface_area / λ^2
    N_edges = Int(round(1.5 * N_triangles))  # Approximate edge count
    memory_gib = 16 * N_edges^2 / 1024^3
    
    return (λ=λ, N_triangles=N_triangles, N_edges=N_edges, memory_gib=memory_gib)
end

# Exercise 2: Scaling verification
function verify_scaling_laws()
    sizes = [0.1, 0.2, 0.4, 0.8]  # Electrical sizes in meters
    results = []
    for a in sizes
        mesh = read_obj_mesh("sphere_$(a).obj")  # Sphere meshes should be created externally or loaded via read_obj_mesh
        rwg = build_rwg(mesh)
        N = rwg.nedges
        push!(results, (a=a, N=N, memory=16*N^2/1024^3))
    end
    return results
end
```

---

## 11. Code Mapping and Further Reading

### 11.1 Relevant Source Files

- **Core type aliases and numeric conventions:** `src/Types.jl`
- **Green's function implementation:** `src/basis/Greens.jl`
- **EFIE operator assembly:** `src/assembly/EFIE.jl`
- **RWG basis functions:** `src/basis/RWG.jl`
- **Linear system solver:** `src/solver/Solve.jl`

### 11.2 Mathematical Prerequisites Review

If you need to refresh the mathematical foundations:
- Vector calculus (gradient, divergence, curl, theorems)
- Complex analysis and phasor notation
- Linear algebra (matrix operations, eigenvalue problems)
- Fourier analysis and spectral methods

See the **Mathematical Prerequisites** appendix for detailed reviews.

### 11.3 Recommended Textbooks

1. **Classical references:**
   - Harrington, R. F. (1993). *Field Computation by Moment Methods*. IEEE Press.
   - Chew, W. C., et al. (2001). *Fast and Efficient Algorithms in Computational Electromagnetics*. Artech House.
   - Gibson, W. C. (2008). *The Method of Moments in Electromagnetics*. Chapman & Hall/CRC.

2. **Modern computational focus:**
   - Peterson, A. F., et al. (1998). *Computational Methods for Electromagnetics*. IEEE Press.
   - Jin, J. (2014). *The Finite Element Method in Electromagnetics*. Wiley.

3. **Specialized on integral equations:**
   - Graglia, R. D., & Lombardi, G. (2004). *Singular Integrals in Integral Equations and Moment Methods*. IEEE Press.

### 11.4 Continuing to Chapter 2

In the next chapter, we will delve deeper into the **EFIE and Boundary Conditions**, where we will:
- Derive the mixed-potential form used in implementation
- Examine the impedance boundary condition in detail
- Discuss numerical implementation details and sign conventions
- Introduce the concept of preconditioning for ill-conditioned systems

---

## 12. Chapter Checklist

Before proceeding, ensure you understand:

- [ ] The derivation of EFIE from Maxwell's equations
- [ ] The physical meaning of surface equivalence principles
- [ ] Why Green's functions automatically satisfy radiation conditions
- [ ] The scaling laws: $N_{\text{surf}} \sim (L/h)^2$ vs $N_{\text{vol}} \sim (L/h)^3$
- [ ] The dense matrix trade-off: fewer unknowns but full coupling
- [ ] How the Method of Moments discretizes integral equations
- [ ] The computational complexity: $O(N^2)$ storage, $O(N^3)$ direct solve

If any items are unclear, review the relevant sections or consult the mathematical prerequisites appendix.
