# Future Development Roadmap for DiffMoM.jl

## Overview
This document outlines the development path to evolve `DiffMoM.jl` from its current specialized differentiable EFIE-MoM framework into a comprehensive general-purpose EM integral equation solver while maintaining its core strength in differentiable inverse design.

## Current Status Assessment (as of February 2026)
The codebase currently provides:
- **Specialization**: Differentiable EFIE-MoM for reactive impedance metasurface design
- **Completeness**: Full forward-adjoint-gradient pipeline with verification
- **Validation**: Internal consistency, dipole/loop analytical gates, Mie benchmarks, Bempp-cl cross-validation
- **Assembly**: Dense O(N²), matrix-free on-demand, and ACA H-matrix O(N log² N) compressed operators
- **Solvers**: Direct LU, GMRES (Krylov.jl) with near-field sparse/diagonal preconditioning, automatic method selection via `solve_scattering`
- **Excitation**: Plane wave, delta gap, port, dipole, loop, imported field, pattern feed, multi-source
- **Mesh tooling**: Quality diagnostics, automatic repair/coarsening, resolution diagnostics, midpoint refinement
- **High-frequency**: Physical Optics (PO) solver for fast RCS reference
- **Scope**: PEC + surface impedance sheets; thread-parallel ACA block assembly

For a general EM MoM solver, the codebase is approximately **45-55% complete** in terms of feature coverage.

## Development Philosophy
1. **Maintain differentiable first**: All new features must support adjoint gradients
2. **Modular design**: Keep physics separate from numerics
3. **Validation-driven**: Every new feature requires comprehensive testing
4. **Performance-awareness**: Scalability for realistic problem sizes

## Phase 1: Foundation Expansion

### 1.1 Equation Formulation Extensions
**Priority: High** | 

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **MFIE** | Magnetic Field Integral Equation for closed PEC bodies | New module `MFIE.jl`, reuse mesh/RWG infrastructure | [ ] |
| **CFIE** | Combined Field Integral Equation (α·EFIE + (1-α)·MFIE) | Combine EFIE/MFIE operators, parameter α | [ ] |
| **PMCHWT** | Poggio-Miller-Chang-Harrington-Wu-Tsai for dielectrics | Volume/surface equivalence, material parameters | [ ] |
| **Generalized IBC** | Anisotropic/impedance tensor support | Extend `Impedance.jl` to matrix-valued Z_s | [ ] |

### 1.2 Material Model Support
**Priority: Medium** | 

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **Dielectric materials** | Complex ε_r, μ_r support | Volume discretization or surface equivalence | [ ] |
| **Layered materials** | Multilayer structures | Generalized impedance/admittance matrices | [ ] |
| **Frequency dispersion** | Lorentz/Drude/Debye models | Complex permittivity functions | [ ] |
| **Anisotropic materials** | Tensor ε, μ | Matrix-valued material parameters | [ ] |

### 1.3 Excitation Extensions
**Priority: High** | 

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **Port excitations** | Waveguide, coax, delta-gap sources | Modal expansion, voltage/current sources | [ ] (basic edge/delta-gap implemented; waveguide/coax modal ports pending) |
| **Near-field sources** | Dipoles, loops, arbitrary current sources | Analytical or imported field patterns | [x] (dipole/loop + analytical CI gates) |
| **Multiple excitations** | Simultaneous multi-source problems | Multi-RHS solves, block operations | [x] |
| **Incident field import** | Measured/numerical incident fields | Field interpolation onto mesh | [x] |
| **Pattern-feed import** | Imported spherical ``E_\theta/E_\phi`` data | Bilinear interpolation + convention handling | [x] |
| **Reflector-feed workflow** | Imported feed + reflector illumination demo | Pattern adapter + reflector mesh example | [x] |

### 1.4 Geometry and Mesh Workflow
**Priority: High** |

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **Mesh quality precheck** | Manifoldness/degeneracy/orientation checks | `mesh_quality_report`, `assert_mesh_quality` | [x] |
| **Automatic mesh repair** | Drop invalid/degenerate triangles, fix orientation | `repair_mesh_for_simulation` / `repair_obj_mesh` | [x] |
| **Automatic mesh coarsening** | Target RWG-count reduction for large OBJ meshes | `coarsen_mesh_to_target_rwg` | [x] |
| **Reflector geometry builder** | Open parabolic reflector mesh for feed studies | `make_parabolic_reflector` | [x] |
| **Resolution diagnostics** | Frequency-based edge-length check (λ/N criterion) | `mesh_resolution_report`, `mesh_resolution_ok` | [x] |
| **Mesh refinement** | Uniform midpoint subdivision to target edge length | `refine_mesh_to_target_edge`, `refine_mesh_for_mom` | [x] |
| **General CAD formats (STEP/IGES)** | STL/MSH native import + gmsh CLI bridge for STEP/IGES | `MeshIO.jl`: `read_stl_mesh`, `read_msh_mesh`, `read_mesh` unified dispatcher, `convert_cad_to_mesh` via gmsh | [x] |

## Phase 2: Computational Scalability 

### 2.1 Fast Algorithms
**Priority: Critical** |

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **Fast Multipole Method** | O(N log N) matrix-vector products | New module `FMM.jl`, tree structures | [ ] |
| **H-Matrix / ACA** | Hierarchical matrices with adaptive cross approximation | `ClusterTree.jl` + `ACA.jl`: binary BSP tree, partially-pivoted ACA, `ACAOperator` with O(N log² N) matvec, thread-parallel block assembly | [x] |
| **Matrix-free operators** | On-the-fly kernel evaluation | `MatrixFreeEFIEOperator` in `EFIE.jl`: `AbstractMatrix` interface, single-entry access, matvec, adjoint | [x] |
| **Physical Optics solver** | PO high-frequency RCS reference | `PhysicalOptics.jl`: `solve_po` for PEC plane-wave scattering, no RWG needed | [x] |
| **GPU acceleration** | CUDA/ROCm support for dense operations | `CUDA.jl`/`AMDGPU.jl` integration | [ ] |

### 2.2 Iterative Solvers
**Priority: High** |

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **GMRES** | Generalized Minimal Residual method | `IterativeSolve.jl` via `Krylov.jl`: left/right preconditioning, forward + adjoint solvers | [x] |
| **Near-field preconditioner** | Sparse NF extraction + LU factorization | `NearFieldPreconditioner.jl`: spatial hashing, sparse LU (`:lu`) or Jacobi diagonal (`:diag`), N-independent iteration counts | [x] |
| **High-level workflow** | Auto method selection based on problem size | `Workflow.jl`: `solve_scattering` dispatches dense direct / dense GMRES / ACA GMRES, auto preconditioner | [x] |
| **Advanced preconditioners** | Calderón, loop-star, sparse approximate inverse | New module | [ ] (mass-based regularization + conditioning helpers in `Solve.jl`) |
| **Deflation** | Deflated GMRES for multiple RHS | Spectral information reuse | [ ] |
| **Krylov recycling** | Recycled Krylov subspaces | For parameter sweeps | [ ] |

### 2.3 Parallel Computation
**Priority: Medium** |

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **Distributed memory** | MPI for large-scale problems | `MPI.jl` integration | [ ] |
| **Thread parallelism** | Multi-threaded assembly and solves | `Threads.@threads`, task parallelism | [partial] (ACA block assembly is `@threads`-parallel; dense EFIE and NF preconditioner are serial) |
| **Hybrid parallelism** | MPI+OpenMP/threads | Hierarchical parallelization | [ ] |
| **Domain decomposition** | For very large problems | Schur complement methods | [ ] |

## Phase 3: Advanced Features 

### 3.1 Shape Derivatives and Geometry Optimization
**Priority: High** | 

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **Parametric shape derivatives** | ∂Z/∂x for vertex motions | Formulation in Section III-G of paper | [ ] |
| **Level set methods** | Implicit shape representation | Level set evolution with adjoint | [ ] |
| **CAD integration** | NURBS/B-spline parameterization | `GeometryBasics.jl` integration | [ ] |
| **Manufacturing constraints** | Minimum feature size, draft angles | Constraint handling in optimization | [ ] |

### 3.2 Frequency Domain Features
**Priority: Medium** | 

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **Broadband analysis** | Frequency sweeps | Adaptive sampling, model order reduction | [ ] |
| **Multi-frequency optimization** | Broadband objectives | Weighted sum over frequency points | [ ] |
| **Asymptotic waveform evaluation** | Fast frequency sweeps | Padé approximation | [ ] |
| **Resonance analysis** | Eigenvalue problems for resonant modes | Arnoldi/Lanczos methods | [ ] |

### 3.3 Antenna and Circuit Integration
**Priority: Medium** | 

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **S-parameter computation** | Port impedance/admittance matrices | Integration with circuit simulators | [ ] |
| **Antenna parameters** | Gain, efficiency, polarization, VSWR | Extended `Diagnostics.jl` | [ ] |
| **Lumped element loading** | RLC components at mesh edges | Generalized impedance loading | [ ] |
| **Nonlinear devices** | Diodes, transistors (harmonic balance) | Nonlinear circuit co-simulation | [ ] |

## Phase 4: Validation 

### 4.1 Comprehensive Validation Suite
**Priority: Critical** | 

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **Benchmark suite** | Standard test cases (sphere, canonical PEC bodies, etc.) | Reference solutions database + CI gates | [x] (sphere-Mie + dipole/loop + pattern-feed + cross-validation workflows) |
| **Measurement validation** | Comparison with experimental data | Import/export utilities | [ ] |
| **Multi-solver validation** | Cross-validation with 3+ other solvers | Automated comparison framework | [ ] (initial Bempp-cl workflow exists; additional solvers pending) |
| **Uncertainty quantification** | Numerical error estimation | Sensitivity to discretization parameters | [ ] |

### 4.2 User Experience and Integration
**Priority: High** | 

| Feature | Description | Implementation Path | Status |
|---------|-------------|-------------------|--------|
| **GUI/Visualization** | Interactive simulation setup and results | `Makie.jl`/`Pluto.jl` integration | [ ] |
| **OBJ/MAT/STL/MSH geometry workflow** | Practical import/repair/coarsen/plot pipeline | `ex_obj_rcs_pipeline.jl` + `MeshIO.jl` (STL, MSH, unified dispatcher) | [x] |
| **CAD import/export** | Standard format support (STEP, IGES) via gmsh CLI bridge | `MeshIO.jl`: `convert_cad_to_mesh` + native STL/MSH readers | [x] |
| **Python interface** | PyCall/Juliacall for Python users | Wrapper generation | [ ] |
| **Cloud deployment** | Web interface for educational use | `Genie.jl` web framework | [ ] |

## Implementation Priorities Matrix

### Must-Have
1. MFIE/CFIE formulations (essential for closed bodies)
2. Fast Multipole Method (scalability to 100k+ unknowns)
3. ~~Iterative solvers with preconditioning~~ **DONE**: GMRES via Krylov.jl + near-field sparse/diagonal preconditioner + auto workflow
4. Dielectric material support (broadens application domain)

### Should-Have 
1. Shape derivatives (unlocks geometry optimization)
2. Port excitations and S-parameters (antenna/RF applications)
3. Parallel computation (multi-core/MPI)
4. Broadband analysis (practical frequency sweeps)

### Nice-to-Have
1. GPU acceleration (maximum performance)
2. Nonlinear device modeling (active circuits)
3. Uncertainty quantification (reliability analysis)
4. Cloud/web interface (accessibility)

## Technical Challenges and Mitigation

### Challenge 1: Maintaining Differentiability
- **Solution**: Abstract operator interface with automatic differentiation fallback
- **Implementation**: `AbstractOperator` type with `apply_forward` and `apply_adjoint` methods

### Challenge 2: Memory Scalability
- **Solution**: Hierarchical data structures with out-of-core options
- **Current**: Native ACA H-matrix (`ACAOperator`) + matrix-free operators reduce memory from O(N²) to O(N log² N). FMM would further improve to O(N log N).

### Challenge 3: Code Complexity Management
- **Solution**: Modular architecture with clear interfaces
- **Implementation**: Well-defined module boundaries, comprehensive tests

### Challenge 4: Validation Burden
- **Solution**: Automated validation pipeline with tolerance thresholds
- **Implementation**: CI/CD with validation matrix, regression testing


## Success Metrics

### Technical Metrics
- Problem size: 1M+ unknowns (with FMM) — **current**: ACA enables ~50k+ unknowns; dense limited to ~10k
- Speed: 10-100x faster than current dense implementation — **current**: ACA + NF preconditioner gives O(N log² N) matvec, N-independent GMRES iters
- Accuracy: < 1% error on standard benchmarks — **current**: Mie sphere validation passing
- Memory: O(N log N) scaling achieved — **current**: O(N log² N) via ACA; O(N²) avoided for large problems


---

## 2026-06 bug hunt — all findings RESOLVED

The deep bug hunt fixed all critical/high correctness bugs (magnetic-dipole factor-i,
PO far-field sign, MLFMA Legendre overflow at L≥90, periodic power balance at oblique
incidence, grounded-EFIE NaN, etc.). A follow-up pass then resolved **all** the
originally-deferred scaling/memory/accuracy items below — each implemented in an
isolated worktree and re-verified against the full suite plus an independent oracle
(bit-identical output for perf items, refined-reference comparison for accuracy items).
The descriptions below are kept as a record of what was addressed.

### Scaling / memory (industrial readiness)
- **radiation_vectors phase cache is `O(NΩ·Nq·Nt)`** (`postprocessing/FarField.jl:81`).
  A dense `ComplexF64` phase cache that does not scale; trade memory for recompute,
  or block over directions, for large far-field grids.
- **PeriodicEFIE ΔG cache is `O(Nq²·Nt²)`** (`assembly/PeriodicEFIE.jl:160`). Dense
  per-pair cache; restructure to stream or exploit translational symmetry.
- **MLFMA translation via `Dict{NTuple{3,Int}}` lookup in the hot matvec**
  (`fast/MLFMA.jl:1444`). Precompute a per-box vector of translation-factor references.
- **EM/DDA dense assembly recomputes each voxel-pair block ~36×** (`mom3d/EMDDA3D.jl`
  dense path) via the generic `Matrix(getindex)` fallback; fill each 6×6 block once.
  (Low priority — large problems use the matrix-free operator.)
- **DDA/EM matvec lacks threading/@inbounds** (`mom3d/DDA3D.jl:274`). Output rows are
  independent (thread-safe), but threading the EM matvec conflicts with its
  `@allocated < 4096` test — needs a per-thread-buffer design.
- **FFT-DDA materializes all 36 kernel blocks at once** (`mom3d/FFTDDA3D.jl:220`);
  reuse one block buffer.
- **DensityFiltering conic filter is `O(Nt²)`** (`optimization/DensityFiltering.jl:47`);
  use spatial binning.
- **Z_corr computes all N² entries even when symmetric at normal incidence**
  (`assembly/PeriodicEFIE.jl:186`).
- **assemble_multiple_excitations / assemble_multi recompute the mesh quadrature cache
  per excitation** (`assembly/Excitation.jl:889,1089`); thread a shared cache through
  the `assemble_*` functions.

### Near-field / operator accuracy (needs a reference to validate a fix)
- **NearField near-singular branch** (`postprocessing/NearField.jl:219`): the
  scalar-potential gradient term uses the full `grad_greens` (1/R² singular,
  un-subtracted) while the vector term is singularity-subtracted; and the vector
  singular part uses only the zeroth-order `J_avg·∫1/R` (omits the linear remainder).
  Both reduce near-surface accuracy; mirror the EFIE self-cell treatment.
- **Dielectric SIE K operator near-singular refinement** only triggers for
  edge-sharing/identical triangle pairs, not vertex-touching pairs
  (`mom3d/SurfaceIE3D.jl:111`).

### Müller SIE — RE-ENABLED ✓
- μ/ε-weighted off-diagonal K + weighted RHS + the second-kind identity (n̂× Gram)
  residue were added in both the dense and matrix-free paths. PMCHWT vs Müller
  currents now agree to <1% (relJ 0.23%, relM 0.68% coarse; 0.12%/0.35% refined,
  tightening) vs ~150% before. Validated by `test/test_surface_ie3d.jl`.

### PTD — general wedge ✓
- The exact `(1/2)tan(γ−v)` reflection-boundary term is now computed for any wedge
  angle γ=nπ (catastrophic-cancellation-safe, reduces to the half-plane n=2 case);
  the interior-wedge warning was removed.

---
*Last updated: June 13, 2026*
