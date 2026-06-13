# API Reference Overview

## Purpose

This page provides a complete function-level map of `DiffMoM.jl`, organized by workflow stage. Use it to locate the right function for your task, then follow the cross-references to detailed documentation.

---

## Import

```julia
using DiffMoM
```

All symbols listed below are exported from the top-level module. No additional `using` or `import` statements are needed for normal usage.

---

## API Map by Workflow

The typical simulation pipeline follows these stages in order. Each stage builds on the outputs of the previous one.

### 1) Geometry and Mesh Safety

Define or import the scatterer geometry and ensure the mesh is suitable for MoM simulation.

- **Types:** `TriMesh`, `RWGData`, `LocalMassMatrix`, `PatchPartition`, `SphGrid`, `ScatteringResult`, `Vec3`, `CVec3`, `AbstractPreconditionerData`, `NearFieldPreconditionerData`, `ILUPreconditionerData`, `DiagonalPreconditionerData`, `BlockDiagPrecondData`, `PermutedPrecondData`
  Core data structures used throughout the package. `LocalMassMatrix` is a compact triplet-stored sparse matrix for per-triangle/per-patch RWG mass blocks. See [types.md](types.md) for field-level documentation.

- **Helpers:** `nvertices`, `ntriangles`
  Quick queries on mesh size.

- **Build/import:** `make_rect_plate`, `make_rect_plate_graded`, `make_circular_plate`, `make_parabolic_reflector`, `read_obj_mesh`, `write_obj_mesh`
  Create meshes programmatically or load from Wavefront OBJ files. `make_circular_plate(radius, Nr, Nphi)` triangulates a disk in the xy-plane using radial rings.

- **Geometry queries:** `triangle_area`, `triangle_center`, `triangle_normal`, `mesh_unique_edges`, `mesh_wireframe_segments`
  Per-triangle geometry and edge extraction.

- **Quality diagnostics:** `mesh_quality_report`, `mesh_quality_ok`, `assert_mesh_quality`
  Check for degenerate triangles, non-manifold edges, and orientation conflicts before assembly.

- **Repair and coarsening:** `repair_mesh_for_simulation`, `repair_obj_mesh`, `coarsen_mesh_to_target_rwg`, `cluster_mesh_vertices`, `drop_nonmanifold_triangles`
  Fix problematic meshes (from CAD exports, etc.) and reduce mesh density to a target RWG count.

- **Resolution diagnostics:** `mesh_resolution_report`, `mesh_resolution_ok`
  Check whether mesh edge lengths satisfy a frequency-based lambda/N criterion. See [mesh.md](mesh.md).

- **Mesh refinement:** `refine_mesh_to_target_edge`, `refine_mesh_for_mom`
  Uniform midpoint subdivision to meet a target maximum edge length or lambda/N criterion. See [mesh.md](mesh.md).

- **Utilities:** `estimate_dense_matrix_gib`
  Estimate memory cost of the dense MoM matrix before assembly.

### 2) Basis Functions and System Assembly

Construct RWG basis functions on the mesh, then assemble the EFIE system matrix and optional impedance loading.

- **RWG basis:** `build_rwg`, `eval_rwg`, `div_rwg`, `basis_triangles`
  Build edge-based Rao-Wilton-Glisson basis functions and evaluate them at any point. See [rwg.md](rwg.md).

- **Green's function kernels:** `greens`, `greens_smooth`, `grad_greens`
  Free-space scalar Green's function and its smooth/gradient variants. Used internally by EFIE assembly.

- **Quadrature:** `tri_quad_rule`, `tri_quad_points`
  Gaussian quadrature rules on the reference triangle.

- **Singular integration:** `analytical_integral_1overR`, `grad_analytical_integral_1overR`, `self_cell_contribution`
  Analytical and hybrid singular integrals for self-cell terms (when source and test triangles overlap). `grad_analytical_integral_1overR` is the closed-form gradient counterpart, used to subtract the `1/R²` singularity of the scalar-potential term near the surface.

- **EFIE assembly:** `assemble_Z_efie`
  Build the dense N x N EFIE impedance matrix. This is the core MoM system matrix for PEC scatterers.

- **Matrix-free EFIE operators:** `matrixfree_efie_operator`, `matrixfree_efie_adjoint_operator`, `efie_entry`
  Matrix-free EFIE matvec without dense N x N allocation. Types: `MatrixFreeEFIEOperator`, `MatrixFreeEFIEAdjointOperator`. See [assembly-solve.md](assembly-solve.md) and [types.md](types.md).

- **Impedance loading:** `precompute_patch_mass`, `assemble_Z_impedance`, `assemble_dZ_dtheta`, `assemble_full_Z`, `assemble_full_Z!`
  Add surface impedance loading for design optimization. `assemble_full_Z!` is the in-place variant that writes `Z(θ) = Z_efie + Z_imp(θ)` into a pre-allocated matrix. See [assembly-solve.md](assembly-solve.md).

- **Periodic kernels and EFIE:** `PeriodicLattice`, `greens_periodic_correction`, `assemble_Z_efie_periodic`
  Build periodic unit-cell operators with Ewald-accelerated Green's correction. See [periodic-methods.md](periodic-methods.md).

- **Composite operator:** `ImpedanceLoadedOperator`
  Matrix-free operator wrapping any `AbstractMatrix{ComplexF64}` base (MLFMA, ACA, dense) with sparse impedance perturbation `Z(theta) = Z_base - Sigma_p theta_p M_p`. Enables GMRES-based optimization with fast operators. See [composite-operators.md](composite-operators.md).

- **Spatial patch assignment:** `assign_patches_grid`, `assign_patches_by_region`, `assign_patches_uniform`, `region_halfspace`, `region_sphere`, `region_box`
  Automatic spatial partitioning of mesh triangles into impedance design patches. See [spatial-patches.md](spatial-patches.md).

### 3) Excitation, Solve, and Field Postprocessing

Apply an incident field, solve for currents, then compute scattered near-field,
total electric field, or far-field observables.

- **Excitation sources:** See [excitation.md](excitation.md) for the full excitation system.
  - Types: `AbstractExcitation`, `PlaneWaveExcitation`, `PortExcitation`, `DeltaGapExcitation`, `DipoleExcitation`, `LoopExcitation`, `MonopoleExcitation`, `ImportedExcitation`, `PatternFeedExcitation`, `MultiExcitation`
  - Constructors: `make_plane_wave`, `make_delta_gap`, `make_dipole`, `make_loop`, `make_monopole`, `make_imported_excitation`, `make_pattern_feed`, `make_analytic_dipole_pattern_feed`, `make_multi_excitation`
  - Assembly: `pattern_feed_field`, `assemble_v_plane_wave`, `assemble_excitation`, `assemble_multiple_excitations`
  - Monopole helpers: `make_monopole(position, axis, height, amplitude, frequency=1e9; include_image=true)` builds a center-fed linear monopole (dipole+image equivalent by default, or a physical half-wire for a meshed ground plane); `monopole_incident_field(r, mono)` evaluates its incident field.
  - Example scripts: `examples/07_pattern_feed.jl`

- **Linear solves (direct):** `solve_forward`, `solve_system`
  Solve the MoM system `Z I = v` using LU factorization (default) or GMRES.

- **Iterative solves (GMRES):** `solve_gmres`, `solve_gmres_adjoint`
  GMRES via Krylov.jl with optional near-field preconditioning. Use these for large problems where direct LU is too slow or memory-intensive.

- **Near-field preconditioner:** `build_nearfield_preconditioner`, `build_block_diag_preconditioner`, `build_mlfma_preconditioner`, `rwg_centers`
  Build sparse near-field preconditioners that dramatically reduce GMRES iteration counts. `build_nearfield_preconditioner` has multiple overloads (dense matrix, abstract matrix, matrix-free operator, geometry/physics, or pre-assembled sparse). Supports sparse LU (`:lu`), incomplete LU (`:ilu`), or Jacobi diagonal (`:diag`) factorization. `build_block_diag_preconditioner` and `build_mlfma_preconditioner` are specialized for MLFMA operators. See [assembly-solve.md](assembly-solve.md) for details and performance data.
  - Types: `AbstractPreconditionerData`, `NearFieldPreconditionerData`, `ILUPreconditionerData`, `DiagonalPreconditionerData`, `BlockDiagPrecondData`, `PermutedPrecondData`, `NearFieldOperator`, `NearFieldAdjointOperator`

- **Far-field computation:** `make_sph_grid`, `radiation_vectors`, `compute_farfield`, `incident_farfield`
  Sample the far-field radiation pattern on a spherical grid. `incident_farfield(excitation, r_hat, k)` returns the asymptotic amplitude of the incident field radiated by an excitation (e.g. `MonopoleExcitation`, `DipoleExcitation`) in direction `r̂`.

- **Near-field computation:** `compute_nearfield`, `compute_total_field`
  Evaluate the scattered electric field, or the total electric field
  `E_total = E_inc + E_sca`, at arbitrary observation points away from the
  surface using the mixed-potential EFIE representation for `E_sca`.

- **Objective (Q-matrix) helpers:** `build_Q`, `build_Q_operator`, `apply_Q`, `pol_linear_x`, `pol_linear_y`, `cap_mask`, `direction_mask`
  Build Hermitian PSD matrices for quadratic far-field objectives used in optimization. `direction_mask` generalizes `cap_mask` to arbitrary directions for multi-angle RCS optimization. `build_Q_operator` returns a matrix-free `FarFieldQMatrix` with the same action as `build_Q` without forming the dense `N x N` matrix.
  - Types: `FarFieldQMatrix` (matrix-free `Q = G' W G` operator), `SumQMatrix` (lazy sum of two same-size Q operators).

### 3b) Fast Methods and High-Level Workflow

For large problems, ACA compression, MLFMA, and the `solve_scattering` workflow automate method selection and preconditioner construction.

- **Cluster tree:** `build_cluster_tree`, `cluster_diameter`, `cluster_distance`, `is_admissible`, `is_leaf`, `leaf_nodes`
  Binary space-partitioning tree for H-matrix block structure. Types: `ClusterNode`, `ClusterTree`. See [aca-workflow.md](aca-workflow.md).

- **ACA low-rank approximation:** `aca_lowrank`, `build_aca_operator`
  Partially-pivoted ACA for far-field block compression. Types: `ACAOperator`, `ACAAdjointOperator`, `DenseBlock` (internal), `LowRankBlock` (internal). See [aca-workflow.md](aca-workflow.md).

- **Octree:** `build_octree`
  Spatial octree decomposition for MLFMA. Types: `Octree`, `OctreeBox`, `OctreeLevel`. See [octree.md](octree.md).

- **MLFMA:** `build_mlfma_operator`, `assemble_mlfma_nearfield`
  Multi-level fast multipole algorithm for O(N log N) matvec. Types: `MLFMAOperator`, `MLFMAAdjointOperator`, `SphereSampling`. See [mlfma.md](mlfma.md).

- **High-level workflow:** `solve_scattering`
  One-call scattering solve with automatic method selection (dense direct, dense GMRES, ACA GMRES, or MLFMA) based on problem size. Returns `ScatteringResult`. See [aca-workflow.md](aca-workflow.md).

### 4) Diagnostics and RCS

Validate simulation results and compute scattering cross sections.

- **Power and conditioning:** `radiated_power`, `projected_power`, `input_power`, `energy_ratio`, `condition_diagnostics`
  Energy-balance checks and matrix conditioning analysis.

- **Radar cross section:** `bistatic_rcs`, `backscatter_rcs`
  Compute bistatic and monostatic RCS from far-field data.

- **Analytical reference:** `mie_s1s2_pec`, `mie_bistatic_rcs_pec`, `mie_s1s2_dielectric`, `mie_bistatic_rcs_dielectric`
  Mie series for PEC and homogeneous dielectric/magnetodielectric sphere scattering; use as a validation reference for your MoM results. The dielectric variants take `eps_r` (and optional `mu_r`) and follow the package-wide `exp(+iωt)` convention.

- **Periodic Floquet metrics:** `FloquetMode`, `floquet_modes`, `reflection_coefficients`, `reflection_coefficient_vectors`, `reflected_power_fractions`, `transmission_coefficients`, `specular_rcs_objective`, `power_balance`
  Post-process periodic unit-cell responses into Floquet coefficients and power accounting. `reflection_coefficient_vectors` returns full (vector) Floquet reflection coefficients and `reflected_power_fractions` gives the per-mode reflected power split. See [periodic-methods.md](periodic-methods.md).

### 4b) Physical Optics and PTD

High-frequency approximate solvers for electrically large problems where full MoM is too expensive.

- **Physical optics solve:** `solve_po`
  Compute PO surface currents and far-field scattering using the tangential magnetic field approximation on illuminated faces. Returns `POResult`. See [physical-optics.md](physical-optics.md).

- **PTD edge diffraction:** `solve_ptd`, `extract_diffraction_edges`
  Physical Theory of Diffraction: adds Ufimtsev fringe corrections from diffraction edges on top of the PO solution. `extract_diffraction_edges` pulls wedge/half-plane edges from a `TriMesh`; `solve_ptd(mesh, freq_hz, excitation; ...)` returns a `PTDResult` with combined PO+PTD, PO-only, and PTD-only far-fields. See [physical-optics.md](physical-optics.md).

- **Types:** `POResult`, `DiffractionEdge`, `PTDResult`
  `POResult` is the result container for PO solutions, analogous to `ScatteringResult` for MoM. `DiffractionEdge` stores the local wedge geometry of one diffraction edge; `PTDResult` is the PTD solver output. See [physical-optics.md](physical-optics.md).

### 4c) Alternative Formulations and Material Solvers

Beyond the PEC surface EFIE, the package provides volume and surface formulations for dielectric and general material scatterers, plus a grounded (half-space) variant of the periodic EFIE.

- **2D volume integral equation (TM):** `assemble_vie_2d`, `solve_vie_2d`, `planewave_2d`, `linesource_2d`, `scattered_field_2d`, `jacobian_scattered_field_2d`, `greens_2d`, `mie_coefficients_2d`, `mie_total_field_2d`
  TM (`E_z`-only) VIE-MoM on a uniform rectangular grid with pulse basis and point matching, for inhomogeneous dielectric domains. Includes plane-wave / line-source excitation, scattered-field evaluation, the contrast Jacobian for differentiable design, and a 2D Mie reference for circular cylinders. Types: `Vec2`, `CVec2`, `Mesh2D`, `VIEResult2D`. See [2D VIE](vie-2d.md).

- **3D volume material solver (DDA / VIE-style):** `make_voxel_grid_3d`, `solve_dda_3d`, `assemble_dda_3d`, `dda_operator_3d`, `planewave_dda_3d`, `scattered_field_dda_3d`, `farfield_dda_3d`
  Discretizes a material volume into a uniform Cartesian voxel grid and solves a vector DDA / volume-integral-equation system for the total fields. The coupled electric-magnetic (bianisotropic) path adds `solve_em_dda_3d`, `assemble_em_dda_3d`, `em_dda_operator_3d`, `planewave_em_dda_3d`, `scattered_fields_em_dda_3d`, `farfield_em_dda_3d`. FFT-accelerated operators (`fft_dda_operator_3d`, `fft_em_dda_operator_3d`) exploit the block-Toeplitz grid structure for fast GMRES; adjoint sensitivities use `solve_dda_adjoint_3d` and `gradient_epsr_dda_3d`. Types: `VoxelGrid3D`, `DDAOperator3D`, `DDAResult3D`, `EMDDAOperator3D`, `EMDDAResult3D`, `FFTDDAOperator3D`, `FFTEMDDAOperator3D`. See [3D Volume DDA](dda-volume-3d.md).

- **3D material models:** `IsotropicMaterial3D`, `DiagonalAnisotropicMaterial3D`, `TensorAnisotropicMaterial3D`, `IsotropicPermeability3D`, `DiagonalPermeability3D`, `TensorPermeability3D`, `MagneticMaterial3D`, `BianisotropicMaterial3D`, `DrudePermittivity3D`, `LorentzPermittivity3D`, `DebyePermittivity3D`
  Constitutive models (relative permittivity, permeability, magnetodielectric and bianisotropic media) consumed by the 3D volume solver, with static and dispersive (Drude / Lorentz / Debye) responses. Evaluators: `material_epsr_3d`, `material_mur_3d`, `material_bianisotropic_matrix_3d`, `drude_epsr_3d`, `lorentz_epsr_3d`, `debye_epsr_3d`. All follow the `exp(+iωt)` convention (passive loss is `imag(eps_r) <= 0`). See [Material Models](material-models-3d.md).

- **Dielectric surface integral equation (3D):** `dielectric_medium_3d`, `assemble_dielectric_sie_3d`, `assemble_pmchwt_3d`, `assemble_muller_3d`, `solve_dielectric_sie_3d`
  Closed-surface SIE for homogeneous isotropic dielectric bodies, solving for tangential `[J; M]` currents via the first-kind PMCHWT or second-kind Müller formulation. Matrix-free assembly is available for GMRES. Types: `DielectricMedium3D`, `DielectricSIEResult3D`, `MatrixFreeDielectricSIE3D`, `MatrixFreeMagneticFieldOperator3D`. Requires a closed mesh (`build_rwg(mesh; allow_boundary=false, require_closed=true)`). See [Dielectric SIE](dielectric-sie-3d.md).

- **Grounded (half-space) EFIE:** `assemble_Z_efie_grounded`, `assemble_excitation_grounded`, `reflection_coefficients_grounded`, `reflection_coefficient_vectors_grounded`
  Image-theory variant of the periodic EFIE for a coplanar metasurface at height `h` above an infinite PEC ground plane: `Z_grounded = Z_direct - Z_image`. Builds on the free-standing periodic EFIE and Floquet post-processing. See [Grounded EFIE](grounded-efie.md) and [periodic-methods.md](periodic-methods.md).

### 5) Differentiable Optimization

Compute gradients via the adjoint method and run impedance optimization.

- **Adjoint primitives:** `compute_objective`, `solve_adjoint`, `solve_adjoint_rhs`, `gradient_impedance`
  The building blocks: evaluate the quadratic objective, solve the adjoint system, and compute the impedance gradient. `solve_adjoint_rhs` accepts a pre-computed RHS for matrix-free Q application or multi-angle objectives. See [adjoint-optimize.md](adjoint-optimize.md).

- **Single-objective optimizers:** `optimize_lbfgs`, `optimize_directivity`
  Projected L-BFGS with box constraints. `optimize_lbfgs` minimizes/maximizes a single quadratic objective; `optimize_directivity` maximizes the ratio of two quadratic objectives (directivity).

- **Multi-angle optimizer:** `optimize_multiangle_rcs`, `build_multiangle_configs`, `AngleConfig`
  Minimize weighted backscatter RCS over multiple incidence angles simultaneously. Supports MLFMA, ACA, and dense base operators via `ImpedanceLoadedOperator`. See [adjoint-optimize.md](adjoint-optimize.md) and the [Multi-Angle RCS chapter](../differentiable-design/05-multiangle-rcs.md).

- **Conditioning helpers:** `make_mass_regularizer`, `make_left_preconditioner`, `select_preconditioner`, `transform_patch_matrices`, `prepare_conditioned_system`
  Advanced: mass-based preconditioning and regularization for ill-conditioned optimization problems.

- **Density topology optimization:** `DensityConfig`, `precompute_triangle_mass`, `assemble_Z_penalty`, `assemble_dZ_drhobar`, `build_filter_weights`, `apply_filter`, `apply_filter_transpose`, `heaviside_project`, `heaviside_derivative`, `filter_and_project`, `gradient_chain_rule`, `gradient_density`, `gradient_density_full`
  End-to-end density interpolation, filtering/projection, and adjoint gradients with respect to raw densities. See [density-topology.md](density-topology.md).

### 6) Verification and Visualization

Check gradient correctness and visualize meshes.

- **Gradient verification:** `complex_step_grad`, `fd_grad`, `verify_gradient`
  Compare adjoint gradients against complex-step and finite-difference references. Essential for validating new objective functions or modified adjoint code. See [verification.md](verification.md).

- **Mesh visualization:** `plot_mesh_wireframe`, `plot_mesh_comparison`, `save_mesh_preview`
  Lightweight 3D wireframe plots for mesh inspection. See [visualization.md](visualization.md).

---

## Recommended Reading Order

For a first read-through of the API documentation, follow this order:

1. **[types.md](types.md)** — Core data structures (`TriMesh`, `RWGData`, `SphGrid`, `ScatteringResult`, preconditioner types, matrix-free operators, etc.)
2. **[mesh.md](mesh.md)** and **[rwg.md](rwg.md)** — Geometry creation, mesh quality, resolution diagnostics, refinement, and RWG basis construction
3. **[assembly-solve.md](assembly-solve.md)** — EFIE assembly (dense and matrix-free), impedance loading, direct/GMRES solvers, and near-field preconditioning
4. **[aca-workflow.md](aca-workflow.md)** — ACA H-matrix compression, cluster trees, and the `solve_scattering` high-level workflow
5. **[octree.md](octree.md)** and **[mlfma.md](mlfma.md)** — Octree spatial decomposition and MLFMA O(N log N) fast solver
6. **[farfield-rcs.md](farfield-rcs.md)** — Near-field, total-field, far-field, Q-matrices, `direction_mask`, RCS, and analytical validation links
7. **[periodic-methods.md](periodic-methods.md)** — `PeriodicLattice`, periodic EFIE assembly, Floquet metrics, and periodic power balance
8. **[composite-operators.md](composite-operators.md)** — `ImpedanceLoadedOperator` for fast-operator optimization
9. **[spatial-patches.md](spatial-patches.md)** — Automatic spatial patch assignment
10. **[adjoint-optimize.md](adjoint-optimize.md)** — Adjoint gradients, L-BFGS optimization, and multi-angle RCS
11. **[density-topology.md](density-topology.md)** — Density interpolation, filtering/projection, and density adjoint gradients
12. **[verification.md](verification.md)** — Gradient correctness checks
13. **[excitation.md](excitation.md)** — Extended excitation system (plane waves, ports, dipoles, monopoles, imported fields, pattern feeds)
14. **[physical-optics.md](physical-optics.md)** — Physical Optics and PTD high-frequency approximate solvers
15. **[vie-2d.md](vie-2d.md)** — 2D TM volume integral equation (VIE) for inhomogeneous dielectric domains
16. **[dda-volume-3d.md](dda-volume-3d.md)** — 3D volume material solver (DDA / EM-DDA / FFT-DDA)
17. **[material-models-3d.md](material-models-3d.md)** — 3D constitutive material models (static and dispersive)
18. **[dielectric-sie-3d.md](dielectric-sie-3d.md)** — Dielectric surface integral equation (PMCHWT / Müller)
19. **[grounded-efie.md](grounded-efie.md)** — Grounded (half-space) periodic EFIE via image theory

---

## Notes on Stability

- All exported function names and signatures listed here are stable for current tutorial and validation workflows.
- Internal helper methods in `src/` may evolve; rely on the exported API for forward compatibility.
