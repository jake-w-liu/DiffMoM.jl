# `solve_scattering` workflow

`solve_scattering` combines mesh validation, RWG construction, method
selection, excitation assembly, preconditioner construction, and the linear
solve. Use it when those defaults match the study. Use the lower-level API when
you need to retain or customize an intermediate operator, basis, or
preconditioner.

Exact keyword defaults are rendered from the source under
[Core docstrings](../api/exported-core.md). The
[ACA and workflow API page](../api/aca-workflow.md) lists the related public
types and functions.

## Minimal solve

```julia
using DiffMoM

frequency = 3.0e9
c0 = 299792458.0
k = 2π * frequency / c0

mesh = make_rect_plate(0.1, 0.1, 8, 8)
source = make_plane_wave(
    Vec3(0.0, 0.0, -k),
    1.0,
    Vec3(1.0, 0.0, 0.0),
)

result = solve_scattering(mesh, frequency, source)
current = result.I_coeffs
println((method=result.method, unknowns=result.N))
```

The excitation can also be a preassembled numeric vector. Its length must
equal the RWG count and every component must be finite.

## Automatic method selection

With `method=:auto`, the workflow compares the RWG count $N$ with three
configurable thresholds:

```math
\operatorname{method}(N)=
\begin{cases}
\texttt{:dense\_direct}, & N\leq N_{\mathrm{direct}},\\
\texttt{:dense\_gmres}, & N_{\mathrm{direct}}<N\leq N_{\mathrm{dense}},\\
\texttt{:aca\_gmres}, & N_{\mathrm{dense}}<N\leq N_{\mathrm{mlfma}},\\
\texttt{:mlfma}, & N>N_{\mathrm{mlfma}}.
\end{cases}
```

The corresponding keywords are `dense_direct_limit`, `dense_gmres_limit`, and
`mlfma_threshold`. They must be nonnegative and ordered. A threshold selects a
code path; it does not prove that the chosen representation fits memory,
converges, or meets the required approximation error.

Force a method only for a measured reason:

```julia
dense = solve_scattering(
    mesh, frequency, source; method=:dense_direct)
aca = solve_scattering(
    mesh, frequency, source; method=:aca_gmres)
```

On a tractable fixed problem, compare the true residual and the final
observable as well as the coefficient vectors. ACA and MLFMA introduce
operator-approximation controls that a small Krylov residual alone cannot
validate.

## Resolution and input checks

The workflow always constructs `mesh_resolution_report`. With
`check_resolution=true`, a mesh that misses the requested
`points_per_wavelength` target produces a warning; setting
`error_on_underresolved=true` turns that warning into an error.

Resolution is a geometric preflight, not an observable-error bound. A usable
result still needs mesh, triangle-quadrature, and observation-grid convergence
for the quantity being reported.

The workflow also rejects:

- non-positive or non-finite frequency and propagation speed;
- unsupported method or preconditioner symbols;
- inconsistent method thresholds;
- a mesh with no RWG unknowns;
- a plane-wave wavenumber or analytic-source frequency inconsistent with the
  supplied frequency and propagation speed; and
- an excitation vector with the wrong length or non-finite components.

The built-in dipole, loop, monopole, and pattern-feed fields use vacuum
`c0=299792458.0`. Use the workflow's default `c0` with those sources. For a
different propagation medium, assemble a consistent excitation vector or use an
imported field rather than combining a vacuum source model with another wave
speed.

## Preconditioners

Preconditioning is used only by the iterative paths.

| Selected method | `preconditioner=:auto` source |
|:--|:--|
| `:dense_gmres` | Sparse near-field entries extracted from the dense matrix |
| `:aca_gmres` | Dense inadmissible ACA blocks |
| `:mlfma` | The MLFMA near-field matrix |

Automatic dense and ACA preconditioning uses sparse LU. Automatic MLFMA
preconditioning uses incomplete LU. `:diag` and `:none` are explicit options;
`:ilu` is accepted only for MLFMA.

`nf_cutoff_lambda` affects the dense-GMRES near-field extraction. ACA
near-field sparsity comes from block admissibility, and MLFMA near-field
sparsity comes from the octree operator, so the cutoff is not their defining
control.

## Convergence checks

For GMRES paths, keep both checks enabled:

- `check_gmres_convergence` rejects failed, inconsistent, or non-finite solver
  results; and
- `check_true_residual` evaluates the returned vector against the selected
  dense, ACA, or MLFMA operator.

The true-residual gate allows `true_residual_factor * gmres_tol`. Both values
must be finite and positive when the check is active. Disable a check only when
the caller deliberately wants a partial iterate and will retain its status and
residual as such.

The direct path uses LU and returns `gmres_iters=-1` and
`gmres_residual=NaN`. For a direct result, compute a physical residual against
the assembled matrix in a lower-level workflow when that evidence is required.

## Result

`ScatteringResult` records:

| Field | Meaning |
|:--|:--|
| `I_coeffs` | Solved RWG coefficient vector |
| `method` | Method actually used |
| `N` | RWG unknown count |
| `assembly_time_s` | Operator assembly time |
| `solve_time_s` | Linear-solve time |
| `preconditioner_time_s` | Preconditioner construction time |
| `gmres_iters` | GMRES iteration count, or `-1` for direct LU |
| `gmres_residual` | Unpreconditioned true relative residual, or `NaN` for direct LU |
| `mesh_report` | Electrical-resolution report |
| `warnings` | Warnings retained by the workflow |

When `method=:auto`, record `result.method` as the effective value. Timing
fields are measurements for that run, not portable performance guarantees.

## When to use the lower-level pipeline

Use explicit calls to `build_rwg`, operator construction, excitation assembly,
preconditioner construction, and solve functions when you need to:

- reuse an RWG basis or operator across several right-hand sides;
- retain a dense matrix for diagnostics or impedance derivatives;
- configure ACA, MLFMA, or preconditioner controls not exposed here;
- measure each intermediate allocation or setup phase independently; or
- compare two methods with precisely shared inputs.

For a frequency sweep, the basis can be reused but the frequency-dependent
operator and excitation must be rebuilt.

## Source map

| Stage | Source |
|:--|:--|
| Workflow and method selection | `src/Workflow.jl` |
| Result type | `src/Types.jl` |
| Mesh and resolution checks | `src/geometry/Mesh.jl` |
| Dense EFIE | `src/assembly/EFIE.jl` |
| ACA | `src/fast/ACA.jl` |
| MLFMA | `src/fast/MLFMA.jl` |
| GMRES and residual checks | `src/solver/IterativeSolve.jl` |
| Near-field preconditioners | `src/solver/NearFieldPreconditioner.jl` |
