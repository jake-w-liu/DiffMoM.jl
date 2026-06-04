# The `solve_scattering` Workflow

## Purpose

A typical MoM scattering analysis requires threading together many separate steps: loading a mesh, building RWG basis functions, assembling the impedance matrix, constructing a preconditioner, solving the linear system, and extracting currents. The `solve_scattering` function wraps the entire pipeline into a single call with sensible defaults and automatic method selection, while still exposing every knob for advanced users who need fine control.

---

## Learning Goals

After this chapter, you should be able to:

1. Explain why a high-level workflow function is valuable alongside the manual pipeline.
2. Describe how `solve_scattering` selects between dense direct, dense GMRES, and ACA GMRES based on problem size.
3. Use mesh resolution validation to catch under-resolved meshes before they produce silently inaccurate results.
4. Call `solve_scattering` with default settings, forced method overrides, and pre-assembled excitation vectors.
5. Interpret the `ScatteringResult` struct and use its timing and diagnostic fields.
6. Decide when to use `solve_scattering` versus the manual step-by-step pipeline.

---

## 1. Motivation: One-Call Scattering Solve

A standard MoM scattering solve requires many manual steps:

```julia
using DiffMoM

mesh = read_obj_mesh("sphere.obj")
rwg  = build_rwg(mesh)
k    = 2pi * freq / 299792458.0
Z    = assemble_Z_efie(mesh, rwg, k)
v    = assemble_excitation(mesh, rwg, pw)
P_nf = build_nearfield_preconditioner(Z, mesh, rwg, cutoff)
I_coeffs, stats = solve_gmres(Z, v; preconditioner=P_nf)
```

Each step requires understanding function signatures, passing the right objects, and choosing appropriate parameters. The `solve_scattering` function collapses this into:

```julia
result = solve_scattering(mesh, 3e9, pw)
I = result.I_coeffs
```

The design philosophy is "batteries included, but removable": sensible defaults for automatic method selection, mesh validation, and preconditioner construction -- with every default overridable via keyword arguments.

---

## 2. Automatic Method Selection

When `method=:auto` (the default), `solve_scattering` selects a solver strategy based on the number of RWG unknowns $N$:

```math
\text{method} =
\begin{cases}
\texttt{:dense\_direct} & \text{if } N \le N_{\text{dd}}, \\
\texttt{:dense\_gmres} & \text{if } N_{\text{dd}} < N \le N_{\text{dg}}, \\
\texttt{:aca\_gmres} & \text{if } N_{\text{dg}} < N \le N_{\text{mlfma}}, \\
\texttt{:mlfma} & \text{if } N > N_{\text{mlfma}},
\end{cases}
```

where $N_{\text{dd}} = 2000$ (`dense_direct_limit`), $N_{\text{dg}} = 10{,}000$ (`dense_gmres_limit`), and $N_{\text{mlfma}} = 50{,}000$ (`mlfma_threshold`) by default.

| N range | Method | Assembly | Solver | Memory |
|---------|--------|----------|--------|--------|
| $N \le 2000$ | `:dense_direct` | `assemble_Z_efie` | LU (`Z \ v`) | $O(N^2)$ |
| $2000 < N \le 10{,}000$ | `:dense_gmres` | `assemble_Z_efie` | GMRES + NF preconditioner | $O(N^2)$ |
| $10{,}000 < N \le 50{,}000$ | `:aca_gmres` | `build_aca_operator` | GMRES + NF preconditioner | $O(N \log^2 N)$ |
| $N > 50{,}000$ | `:mlfma` | `build_mlfma_operator` | GMRES + NF preconditioner from `A_mlfma.Z_near` (auto uses `:ilu`) | $O(N \log N)$ |

The dense $N \times N$ `ComplexF64` matrix occupies $16 N^2$ bytes. At $N = 2000$ this is 61 MiB (LU is fastest). At $N = 10{,}000$ it grows to 1.5 GiB (GMRES avoids $O(N^3)$ LU cost). ACA H-matrix compression (used for $10k < N \le 50k$) reduces storage and matvec to $O(N \log^2 N)$. For very large problems ($N > 50{,}000$), MLFMA achieves $O(N \log N)$ complexity (see Chapter 6: MLFMA).

The user can override auto-selection:

```julia
result = solve_scattering(mesh, freq, pw; method=:aca_gmres)   # force ACA
result = solve_scattering(mesh, freq, pw; method=:dense_direct) # force direct
```

---

## 3. Mesh Resolution Validation

An under-resolved mesh -- where the longest edge exceeds $\lambda / n_{\text{ppw}}$ -- produces solutions that converge to the wrong answer without obvious errors. Before assembly, `solve_scattering` checks:

```math
h_{\max} \le \frac{\lambda}{n_{\text{ppw}}},
```

where $h_{\max}$ is the longest mesh edge, $\lambda = c_0 / f$, and $n_{\text{ppw}} = 10$ by default.

| Setting | Behavior on under-resolution |
|---------|------------------------------|
| `check_resolution=true` (default) | Prints a warning, continues solving |
| `error_on_underresolved=true` | Throws an error, halts execution |
| `check_resolution=false` | Skips the check entirely |

```julia
# Strict mode: fail on bad meshes
result = solve_scattering(mesh, freq, pw; error_on_underresolved=true)

# Skip check for convergence studies with intentionally coarse meshes
result = solve_scattering(mesh, freq, pw; check_resolution=false)
```

---

## 4. The `solve_scattering` Pipeline

Internally, `solve_scattering` executes these steps in order:

1. **Validate inputs**: `freq_hz > 0`, `method` is valid.
2. **Mesh resolution check**: calls `mesh_resolution_report(mesh, freq_hz)`, optionally warns/errors.
3. **Build RWG**: `build_rwg(mesh)` -- the resulting $N$ drives method selection.
4. **Select method**: based on $N$ and thresholds (or user override).
5. **Assemble excitation**: accepts an `AbstractExcitation` object or a pre-assembled `Vector{ComplexF64}`. If given an excitation object, calls `assemble_excitation(mesh, rwg, excitation)`.
6. **Impedance assembly**:
   - Dense: `assemble_Z_efie(mesh, rwg, k; mesh_precheck=false)` -- precheck already done.
   - ACA: `build_aca_operator(mesh, rwg, k; leaf_size, eta, aca_tol, max_rank)`.
7. **Preconditioner** (GMRES methods only):
   - `preconditioner=:auto` defaults to `:lu` for dense/ACA GMRES, and to `:ilu` for MLFMA.
   - Dense GMRES: `build_nearfield_preconditioner(Z, mesh, rwg, cutoff)` -- extracts from dense matrix.
   - ACA GMRES: `build_nearfield_preconditioner(mesh, rwg, k, cutoff)` -- assembles from geometry.
   - MLFMA: `build_nearfield_preconditioner(A_mlfma.Z_near; factorization=...)`.
   - Cutoff distance = `nf_cutoff_lambda * lambda`.
8. **Solve**: direct uses `Z \ v`; GMRES uses `solve_gmres(Z_or_A, v; preconditioner=P_nf, tol, maxiter)`.
9. **Return** a `ScatteringResult` with solution, timing, and diagnostics.

---

## 5. The `ScatteringResult` Struct

```julia
struct ScatteringResult
    I_coeffs::Vector{ComplexF64}     # Surface current coefficients
    method::Symbol                    # :dense_direct, :dense_gmres, :aca_gmres, or :mlfma
    N::Int                           # Number of RWG unknowns
    assembly_time_s::Float64         # Assembly wall time
    solve_time_s::Float64            # Solve wall time
    preconditioner_time_s::Float64   # Preconditioner build time
    gmres_iters::Int                 # GMRES iterations (-1 for direct)
    gmres_residual::Float64          # Final residual (NaN for direct)
    mesh_report::NamedTuple          # From mesh_resolution_report
    warnings::Vector{String}         # Any warnings generated
end
```

| Field | Description | Notes |
|-------|-------------|-------|
| `I_coeffs` | RWG current expansion coefficients | Pass to `compute_farfield`, `bistatic_rcs`, etc. |
| `method` | Solver method actually used | May differ from `:auto` input |
| `assembly_time_s` | Wall time for impedance assembly | Includes ACA compression if applicable |
| `solve_time_s` | Wall time for linear solve | Includes GMRES iterations |
| `gmres_iters` | Number of GMRES iterations | `-1` for direct solves |
| `mesh_report` | Full mesh resolution report | Edge statistics, wavelength, `meets_target` flag |

---

## 6. Keyword Arguments Reference

### Method Selection

| Keyword | Default | Description |
|---------|---------|-------------|
| `method` | `:auto` | One of `:auto`, `:dense_direct`, `:dense_gmres`, `:aca_gmres`, `:mlfma` |
| `dense_direct_limit` | `2000` | $N$ threshold: below this, auto selects dense direct |
| `dense_gmres_limit` | `10000` | $N$ threshold: above this, auto selects ACA GMRES |

### Mesh Validation

| Keyword | Default | Description |
|---------|---------|-------------|
| `check_resolution` | `true` | Run mesh resolution check before solving |
| `points_per_wavelength` | `10.0` | Target mesh density ($n_{\text{ppw}}$) |
| `error_on_underresolved` | `false` | Throw error (vs. warning) on under-resolved mesh |

### Solver Settings

| Keyword | Default | Description |
|---------|---------|-------------|
| `gmres_tol` | `1e-6` | GMRES relative tolerance |
| `gmres_maxiter` | `300` | Maximum GMRES iterations |

### Preconditioner

| Keyword | Default | Description |
|---------|---------|-------------|
| `nf_cutoff_lambda` | `1.0` | Near-field cutoff distance in wavelengths |
| `preconditioner` | `:auto` | `:auto` maps to `:lu` (dense/ACA) or `:ilu` (MLFMA). Explicit `:ilu` is meaningful in the MLFMA path; dense/ACA use `:lu`/`:diag`/`:none`. |

### ACA Settings

| Keyword | Default | Description |
|---------|---------|-------------|
| `aca_tol` | `1e-6` | ACA low-rank approximation tolerance |
| `aca_leaf_size` | `64` | Cluster tree leaf size |
| `aca_eta` | `1.5` | Admissibility parameter ($\eta$) |
| `aca_max_rank` | `50` | Maximum rank per low-rank block |

### General

| Keyword | Default | Description |
|---------|---------|-------------|
| `verbose` | `true` | Print progress information to stdout |
| `quad_order` | `3` | Quadrature order for EFIE entry evaluation |
| `c0` | `299792458.0` | Speed of light in m/s |

---

## 7. Practical Usage Patterns

### 7.1 Basic Usage

```julia
using DiffMoM

mesh = read_obj_mesh("plate.obj")
freq = 3e9
k = 2pi * freq / 299792458.0
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))

result = solve_scattering(mesh, freq, pw)
I = result.I_coeffs
```

### 7.2 Pre-Assembled Excitation

```julia
rwg = build_rwg(mesh)
v = assemble_excitation(mesh, rwg, pw)
result = solve_scattering(mesh, freq, v)
```

### 7.3 Method Comparison for Validation

```julia
result_dense = solve_scattering(mesh, freq, pw; method=:dense_direct)
result_aca   = solve_scattering(mesh, freq, pw; method=:aca_gmres)
err = norm(result_dense.I_coeffs - result_aca.I_coeffs) / norm(result_dense.I_coeffs)
println("ACA relative error: ", err)
```

### 7.4 Inspecting Performance

```julia
result = solve_scattering(mesh, freq, pw)
println("Method:   ", result.method)
println("N:        ", result.N)
println("Assembly: ", result.assembly_time_s, " s")
println("Solve:    ", result.solve_time_s, " s")
if result.method != :dense_direct
    println("GMRES iters: ", result.gmres_iters)
    println("Precond:     ", result.preconditioner_time_s, " s")
end
```

### 7.5 Post-Processing with Far Field and RCS

```julia
result = solve_scattering(mesh, freq, pw)
rwg = build_rwg(mesh)  # needed for far-field computation

grid = make_sph_grid(91, 361)
G_mat = radiation_vectors(mesh, rwg, grid, k)
E_ff = compute_farfield(G_mat, result.I_coeffs, length(grid.w))
rcs = bistatic_rcs(E_ff; E0=1.0)
```

---

## 8. When to Use `solve_scattering` vs. the Manual Pipeline

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Quick scattering analysis | `solve_scattering` | Automatic everything, minimal code |
| Frequency sweep | Manual pipeline | Reuse `mesh`/`rwg`; only reassemble `Z` and `v` |
| Impedance optimization | Manual pipeline | Need `Z`, `Mp`, `Q` individually for adjoint |
| Custom excitation assembly | Pass pre-assembled `v` | Still benefit from auto method selection |
| Debugging / validation | Manual pipeline | Full control over each step |
| ACA accuracy benchmarking | `solve_scattering` with forced method | Two calls with different methods |

For a frequency sweep, the manual pipeline avoids redundant `build_rwg` and resolution checks:

```julia
mesh = read_obj_mesh("antenna.obj")
rwg = build_rwg(mesh)

for freq in range(1e9, 5e9; length=50)
    k = 2pi * freq / 299792458.0
    Z = assemble_Z_efie(mesh, rwg, k)
    v = assemble_excitation(mesh, rwg, make_plane_wave(...))
    I = Z \ v
    # ... post-process
end
```

---

## 9. Code Mapping

| Concept | Source file | Key function / type |
|---------|------------|---------------------|
| High-level workflow | `src/Workflow.jl` | `solve_scattering` |
| Result container | `src/Types.jl` | `ScatteringResult` |
| Mesh resolution check | `src/geometry/Mesh.jl` | `mesh_resolution_report` |
| Dense memory estimate | `src/geometry/Mesh.jl` | `estimate_dense_matrix_gib` |
| RWG construction | `src/basis/RWG.jl` | `build_rwg` |
| Dense EFIE assembly | `src/assembly/EFIE.jl` | `assemble_Z_efie` |
| ACA H-matrix assembly | `src/fast/ACA.jl` | `build_aca_operator` |
| NF preconditioner (from Z) | `src/solver/NearFieldPreconditioner.jl` | `build_nearfield_preconditioner(Z, mesh, rwg, cutoff)` |
| NF preconditioner (geometry) | `src/solver/NearFieldPreconditioner.jl` | `build_nearfield_preconditioner(mesh, rwg, k, cutoff)` |
| GMRES solve | `src/solver/IterativeSolve.jl` | `solve_gmres` |
| Excitation assembly | `src/assembly/Excitation.jl` | `assemble_excitation` |

---

## 10. Exercises

### 10.1 Conceptual Questions

1. **Threshold reasoning**: Explain why the default `dense_direct_limit` is 2000 and not 500 or 50,000. What hardware constraints determine the ideal threshold?

2. **Preconditioner source**: For `:dense_gmres`, the near-field preconditioner is built from the already-assembled dense matrix `Z`. For `:aca_gmres`, the workflow builds it from geometry directly. Why might this be preferable to extracting entries through `A_aca[i,j]`?

3. **Under-resolution risk**: A user sets `check_resolution=false` on a mesh where $h_{\max} = 0.5\lambda$. The solve succeeds. Why might the result be dangerously wrong?

### 10.2 Coding Exercises

1. Use `solve_scattering` to compute the bistatic RCS of a $0.5\lambda \times 0.5\lambda$ PEC plate at 3 GHz. Print the method selected, assembly time, and solve time.

2. For a mesh with $N \approx 3000$, call `solve_scattering` with `:dense_direct`, `:dense_gmres`, and `:aca_gmres`. Then force `:mlfma` on a larger mesh. Compare timing and relative error in `I_coeffs`.

3. Set `dense_direct_limit=500` and `dense_gmres_limit=3000`. Verify that auto-selection changes for $N = 1500$.

4. Create a deliberately coarse mesh and call `solve_scattering` with `error_on_underresolved=true`. Confirm it throws. Relax `points_per_wavelength=5.0` and verify success.

### 10.3 Advanced Challenge

Write a frequency sweep (1--10 GHz, 20 points) using both `solve_scattering` and the manual pipeline. Compare total wall time and explain the overhead difference.

---

## 11. Chapter Checklist

- [ ] Call `solve_scattering` with default settings and extract current coefficients.
- [ ] Explain automatic method selection and the role of `dense_direct_limit` / `dense_gmres_limit` / `mlfma_threshold`.
- [ ] Override method selection with the `method` keyword.
- [ ] Pass either an excitation object or a pre-assembled vector.
- [ ] Interpret all fields of `ScatteringResult`, including timing and GMRES diagnostics.
- [ ] Enable strict mesh validation with `error_on_underresolved=true`.
- [ ] Choose between `solve_scattering` and the manual pipeline based on the use case.
- [ ] Configure preconditioner type and cutoff distance for GMRES solves.

---

## 12. Further Reading

- **Gibson, W. C.** *The Method of Moments in Electromagnetics* (2008) -- Ch. 3 on practical MoM workflow: mesh resolution, solver selection, and post-processing.
- **DiffMoM.jl API docs**: `api/aca-workflow.md` for ACA H-matrix details; `api/assembly-solve.md` for dense assembly and solver dispatch.
- **Saad, Y.** *Iterative Methods for Sparse Linear Systems* (2003) -- Ch. 9 on preconditioning strategies relevant to the near-field preconditioner.

---

*Next: [Multi-Level Fast Multipole Algorithm (MLFMA)](06-mlfma.md) introduces the large-scale hierarchical acceleration path used when `solve_scattering` selects `method=:mlfma`.*
