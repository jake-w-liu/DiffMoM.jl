# Appendix: Symbol Glossary

## Purpose

Quick reference for mathematical symbols used throughout `DiffMoM.jl` documentation and code. Organized by physical domain and mathematical category.

---

## Fundamental Physical Constants

| Symbol | Meaning | Typical Value | Units |
|--------|---------|---------------|-------|
| $c_0$ | Speed of light in vacuum | $2.99792458\times10^8$ | m/s |
| $\mu_0$ | Vacuum permeability | $4\pi\times10^{-7}$ | H/m |
| $\epsilon_0$ | Vacuum permittivity | $8.854187817\times10^{-12}$ | F/m |
| $\eta_0$ | Free-space impedance | $\sqrt{\mu_0/\epsilon_0}\approx376.73$ | Ω |
| $\pi$ | Mathematical constant | $3.141592653589793$ | - |

---

## Frequency and Wave Parameters

| Symbol | Meaning | Formula | Units |
|--------|---------|---------|-------|
| $f$ | Frequency | - | Hz |
| $\omega$ | Angular frequency | $\omega=2\pi f$ | rad/s |
| $\lambda_0$ | Wavelength | $\lambda_0=c_0/f$ | m |
| $k$ | Wavenumber | $k=2\pi/\lambda_0=\omega/c_0$ | rad/m |
| $T$ | Time period | $T=1/f$ | s |

---

## Geometry and Coordinates

| Symbol | Meaning | Description | Units |
|--------|---------|-------------|-------|
| $\Gamma$ | Scattering surface | Surface of PEC/impedance object | m² |
| $\mathbf r$ | Observation point | Position vector where field is evaluated | m |
| $\mathbf r'$ | Source point | Position vector where current resides | m |
| $\hat{\mathbf n}$ | Unit surface normal | Outward normal to surface `Γ` | - |
| $\hat{\mathbf t}$ | Unit tangent vector | Along surface boundary | - |
| $\mathbf P_t$ | Tangential projection | $\mathbf P_t=\mathbf I-\hat{\mathbf n}\hat{\mathbf n}^T$ | - |
| $\theta$ | Polar angle | From +z axis, $\theta\in[0,\pi]$ | rad |
| $\phi$ | Azimuthal angle | In x-y plane, $\phi\in[0,2\pi)$ | rad |
| $R$ | Distance | $R=\lvert\mathbf r-\mathbf r'\rvert$ | m |

---

## RWG Basis Functions (Discretization)

| Symbol | Meaning | Description |
|--------|---------|-------------|
| $N$ | Number of RWG unknowns | Equal to number of interior edges |
| $N_v$ | Number of vertices | Mesh vertex count |
| $N_t$ | Number of triangles | Mesh triangle count |
| $\mathbf f_n(\mathbf r)$ | RWG basis function | Linear vector function on triangles `T±` |
| $\ell_n$ | Edge length | Length of shared interior edge | m |
| $A_n^+$, $A_n^-$ | Triangle areas | Areas of `T+` and `T-` triangles | m² |
| $\mathbf r_{+,\mathrm{opp}}$ | Opposite vertex in `T+` | Vertex not on shared edge in `T+` |
| $\mathbf r_{-,\mathrm{opp}}$ | Opposite vertex in `T-` | Vertex not on shared edge in `T-` |
| $\nabla_s\cdot$ | Surface divergence operator | Divergence within surface plane |
| $\mathbf J(\mathbf r)$ | Surface current density | $\mathbf J(\mathbf r)=\sum_n I_n\mathbf f_n(\mathbf r)$ | A/m |

---

## Linear System and Operators

| Symbol | Meaning | Formula/Description |
|--------|---------|---------------------|
| $\mathbf Z$ | MoM impedance matrix | $Z_{mn}=\langle\mathbf f_m,\mathcal T[\mathbf f_n]\rangle$ |
| $\mathbf v$ | Excitation vector | $v_m=-\langle\mathbf f_m,\mathbf E_t^{\mathrm{inc}}\rangle$ |
| $\mathbf I$ | Current coefficient vector | $\mathbf I\in\mathbb C^N$ |
| $\mathcal T$ | EFIE operator | $\mathcal T[\mathbf J]=-i\omega\mu_0\int_\Gamma[\bar{\mathbf I}+\frac{1}{k^2}\nabla\nabla\cdot]G\mathbf J\,dS'$ |
| $G(\mathbf r,\mathbf r')$ | Green's function | $G=e^{-ikR}/(4\pi R)$ |
| $\mathbf M_p$ | Patch mass matrix | $(M_p)_{mn}=\langle\mathbf f_m,\mathbf f_n\rangle$ over patch `p` |
| $\mathbf Z_{\mathrm{imp}}$ | Impedance matrix | $\mathbf Z_{\mathrm{imp}}=-\sum_p\theta_p\mathbf M_p$ |
| $\mathbf Z_\alpha$ | Regularized system | $\mathbf Z_\alpha=\mathbf Z+\alpha\mathbf R$ |

---

## Field Quantities

| Symbol | Meaning | Description | Units |
|--------|---------|-------------|-------|
| $\mathbf E^{\mathrm{inc}}$ | Incident electric field | Known excitation field | V/m |
| $\mathbf E^{\mathrm{sca}}$ | Scattered electric field | Field radiated by surface currents | V/m |
| $\mathbf E^{\mathrm{tot}}$ | Total electric field | $\mathbf E^{\mathrm{inc}}+\mathbf E^{\mathrm{sca}}$ | V/m |
| $\mathbf E_t$ | Tangential electric field | $\mathbf P_t\mathbf E$ | V/m |
| $\mathbf E^\infty$ | Far-field amplitude | Asymptotic field as $r\to\infty$ | V |
| $\mathbf H$ | Magnetic field | - | A/m |

---

## Optimization and Adjoint Method

| Symbol | Meaning | Description |
|--------|---------|-------------|
| $J$ | Objective function | Scalar quantity to optimize |
| $\mathbf Q$ | Quadratic objective matrix | $J=\mathbf I^\dagger\mathbf Q\mathbf I$ |
| $\mathbf Q_t$ | Target region matrix | Numerator in ratio objective |
| $\mathbf Q_{\mathrm{tot}}$ | Total region matrix | Denominator in ratio objective |
| $\boldsymbol\lambda$ | Adjoint vector | Solution of $\mathbf Z^\dagger\boldsymbol\lambda=\partial J/\partial\mathbf I^*$ |
| $\boldsymbol\theta$ | Design parameter vector | $\boldsymbol\theta\in\mathbb R^P$ |
| $\partial J/\partial\theta_p$ | Parameter sensitivity | Gradient w.r.t. parameter `p` |
| $\alpha$ | Regularization parameter | Small positive scalar |

---

## Far-Field and Radar Cross Section

| Symbol | Meaning | Formula | Units |
|--------|---------|---------|-------|
| $\sigma$ | Radar Cross Section | $\sigma=4\pi r^2\lvert\mathbf E^{\mathrm{sca}}\rvert^2/\lvert\mathbf E^{\mathrm{inc}}\rvert^2$ | m² |
| $\hat{\mathbf r}$ | Observation direction | Unit vector $(\theta,\phi)$ | - |
| $\Delta\theta,\Delta\phi$ | Angular grid spacing | Discretization steps | rad |
| $w_q$ | Quadrature weight | $w_q=\sin\theta_q\Delta\theta\Delta\phi$ | sr |

---

## Mathematical Operators and Notation

| Symbol | Meaning | Usage |
|--------|---------|-------|
| $\nabla$ | Gradient operator | $\nabla\phi$ |
| $\nabla\cdot$ | Divergence operator | $\nabla\cdot\mathbf F$ |
| $\nabla\times$ | Curl operator | $\nabla\times\mathbf F$ |
| $\langle\cdot,\cdot\rangle$ | Inner product | $\langle\mathbf f,\mathbf g\rangle=\int_\Gamma\mathbf f\cdot\mathbf g^*\,dS$ |
| $(\cdot)^*$ | Complex conjugate | $a^*$ |
| $(\cdot)^T$ | Transpose | $\mathbf A^T$ |
| $(\cdot)^\dagger$ | Conjugate transpose | $\mathbf A^\dagger=(\mathbf A^*)^T$ |
| $\|\cdot\|$ | Norm | $\|\mathbf v\|=\sqrt{\mathbf v^\dagger\mathbf v}$ |
| $\text{Re}\{\cdot\}$ | Real part | $\text{Re}\{a+ib\}=a$ |
| $\text{Im}\{\cdot\}$ | Imaginary part | $\text{Im}\{a+ib\}=b$ |
| $i$ | Imaginary unit | $i=\sqrt{-1}$ |

---

## Validation and Diagnostics

| Symbol | Meaning | Description |
|--------|---------|-------------|
| $\kappa(\mathbf Z)$ | Condition number | $\|\mathbf Z\|\cdot\|\mathbf Z^{-1}\|$ |
| $P_{\mathrm{rad}}$ | Radiated power | $P_{\mathrm{rad}}=\frac{1}{2\eta_0}\int \lvert\mathbf E^\infty\rvert^2\,d\Omega$ | W |
| $P_{\mathrm{in}}$ | Input power | $P_{\mathrm{in}}=-\frac{1}{2}\text{Re}\{\mathbf I^\dagger\mathbf v\}$ | W |
| FD | Finite Difference | Numerical gradient approximation |
| CS | Complex Step | Exact gradient via complex perturbation |
| RMSE | Root Mean Square Error | $\sqrt{\frac{1}{N}\sum_i\lvert y_i-\hat{y}_i\rvert^2}$ |
| SLL | Sidelobe Level | Ratio of main lobe to highest sidelobe | dB |

---

## Code Mapping

- Core types and vector aliases: `src/Types.jl`
- Geometry and mesh operations: `src/geometry/Mesh.jl`
- RWG basis: `src/basis/RWG.jl`
- Green's function: `src/basis/Greens.jl`
- Matrix assembly: `src/assembly/EFIE.jl`, `src/assembly/Impedance.jl`
- Optimization: `src/optimization/Adjoint.jl`, `src/optimization/Optimize.jl`
