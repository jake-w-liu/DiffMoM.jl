# Workflow.jl — High-level scattering solve with automatic method selection
#
# Provides `solve_scattering` which validates mesh resolution, selects the
# solver method (dense direct / dense GMRES / ACA GMRES / MLFMA)
# based on problem size, and handles preconditioner setup automatically.

export solve_scattering

const C0_DEFAULT = 299792458.0

function _workflow_dense_direct_work_bytes(system_size::Int)
    field_payload = _checked_array_payload_bytes(
        ComplexF64, 2, system_size;
        label="solve_scattering direct field vectors")
    return _checked_dense_lu_work_bytes(
        ComplexF64,
        system_size,
        2,
        field_payload;
        label="solve_scattering dense direct solve",
    )
end

function _workflow_exact_dense_direct_work_bytes(system_size::Int)
    # The physical matrix, rejected IEEE factor/equilibration retries, input
    # and output vectors can coexist with the bounded exact factor and solve.
    return _checked_exact_dense_solve_work_bytes(
        ComplexF64,
        system_size,
        3,
        3,
        3;
        label="solve_scattering exact dense direct solve",
    )
end

"""
    solve_scattering(mesh, freq_hz, excitation; kwargs...)

High-level scattering solve that selects a method from the configured problem-
size thresholds:

- **N <= dense_direct_limit** (default 2000): Dense EFIE assembly + LU direct solve
- **dense_direct_limit < N <= dense_gmres_limit** (default 10000): Dense + NF-preconditioned GMRES
- **dense_gmres_limit < N <= mlfma_threshold** (default 50000): ACA H-matrix + NF-preconditioned GMRES
- **N > mlfma_threshold**: MLFMA O(N log N) + NF-preconditioned GMRES

Mesh resolution is validated against the frequency. Under-resolved meshes
produce a warning (or error if `error_on_underresolved=true`).

# Arguments
- `mesh::TriMesh`: triangle surface mesh
- `freq_hz::Real`: frequency in Hz
- `excitation`: either an `AbstractExcitation` or a pre-assembled
  `Vector{ComplexF64}` excitation vector. Frequency-bearing excitations must
  match `freq_hz` and `c0`; built-in dipole, loop, monopole, and pattern-feed
  source models use vacuum `c0`.

# Keyword Arguments
## Method selection
- `method=:auto`: one of `:auto`, `:dense_direct`, `:dense_gmres`, `:aca_gmres`, `:mlfma`
- `dense_direct_limit=2000`: N threshold below which dense direct is used
- `dense_gmres_limit=10000`: N threshold below which dense GMRES is used (above → ACA)
- `mlfma_threshold=50000`: N threshold above which MLFMA is used instead of ACA

## Mesh validation
- `check_resolution=true`: run mesh resolution check
- `points_per_wavelength=10.0`: target mesh density
- `error_on_underresolved=false`: throw error instead of warning

## Solver settings
- `gmres_tol=1e-6`: GMRES relative tolerance
- `gmres_maxiter=300`: maximum GMRES iterations
- `check_gmres_convergence=true`: reject unconverged or non-finite GMRES results
- `check_true_residual=true`: verify the residual against the selected operator
- `true_residual_factor=100.0`: allowed true-residual multiple of `gmres_tol`

## NF preconditioner
- `nf_cutoff_lambda=1.0`: near-field cutoff in wavelengths for dense GMRES;
  ACA and MLFMA use their stored near-field structures
- `preconditioner=:auto`: one of `:auto`, `:lu`, `:ilu` (MLFMA), `:diag`, `:none`

## ACA settings
- `aca_tol=1e-6`: ACA low-rank approximation tolerance
- `aca_leaf_size=64`: cluster tree leaf size
- `aca_eta=1.5`: admissibility parameter
- `aca_max_rank=50`: maximum rank per low-rank block

## General
- `verbose=true`: print progress info
- `quad_order=3`: quadrature order for EFIE entries
- `c0=299792458.0`: speed of light (m/s)
- `max_dense_matrix_bytes=2_000_000_000`: raw-payload ceiling for the dense
  EFIE matrix, and for the simultaneous matrix, factor, pivot, and field
  buffers on the dense-direct path

# Returns
A `ScatteringResult` with fields: `I_coeffs`, `method`, `N`, timing info,
GMRES stats, `mesh_report`, and `warnings`. For iterative methods,
`gmres_residual` is the unpreconditioned true relative residual against the
selected operator; it is `NaN` for a direct solve.
"""
function solve_scattering(mesh::TriMesh, freq_hz::Real, excitation;
                          method::Symbol=:auto,
                          dense_direct_limit::Int=2000,
                          dense_gmres_limit::Int=10000,
                          mlfma_threshold::Int=50000,
                          check_resolution::Bool=true,
                          points_per_wavelength::Real=10.0,
                          error_on_underresolved::Bool=false,
                          gmres_tol::Float64=1e-6,
                          gmres_maxiter::Int=300,
                          check_gmres_convergence::Bool=true,
                          check_true_residual::Bool=true,
                          true_residual_factor::Float64=100.0,
                          nf_cutoff_lambda::Float64=1.0,
                          preconditioner::Symbol=:auto,
                          aca_tol::Float64=1e-6,
                          aca_leaf_size::Int=64,
                          aca_eta::Float64=1.5,
                          aca_max_rank::Int=50,
                          verbose::Bool=true,
                          quad_order::Int=3,
                          c0::Real=C0_DEFAULT,
                          max_dense_matrix_bytes::Integer=
                              _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    frequency = Float64(freq_hz)
    isfinite(frequency) && frequency > 0 ||
        throw(ArgumentError(
            "solve_scattering: freq_hz must be finite and positive, got $freq_hz"))
    propagation_speed = Float64(c0)
    isfinite(propagation_speed) && propagation_speed > 0 ||
        throw(ArgumentError(
            "solve_scattering: c0 must be finite and positive, got $c0"))
    method in (:auto, :dense_direct, :dense_gmres, :aca_gmres, :mlfma) ||
        error(
            "solve_scattering: method must be :auto, :dense_direct, " *
            ":dense_gmres, :aca_gmres, or :mlfma")
    preconditioner in (:auto, :lu, :ilu, :diag, :none) ||
        throw(ArgumentError(
            "solve_scattering: preconditioner must be :auto, :lu, :ilu, " *
            ":diag, or :none"))
    dense_direct_limit >= 0 ||
        throw(ArgumentError(
            "solve_scattering: dense_direct_limit must be nonnegative, got $dense_direct_limit"))
    dense_gmres_limit >= dense_direct_limit ||
        throw(ArgumentError(
            "solve_scattering: dense_gmres_limit ($dense_gmres_limit) must be at least " *
            "dense_direct_limit ($dense_direct_limit)"))
    mlfma_threshold >= dense_gmres_limit ||
        throw(ArgumentError(
            "solve_scattering: mlfma_threshold ($mlfma_threshold) must be at least " *
            "dense_gmres_limit ($dense_gmres_limit)"))
    if check_true_residual
        (isfinite(true_residual_factor) && true_residual_factor > 0.0) ||
            throw(ArgumentError(
                "solve_scattering: true_residual_factor must be finite and " *
                "positive, got $true_residual_factor"))
    end

    warnings = String[]
    lambda = propagation_speed / frequency
    k = _frequency_to_wavenumber(
        frequency, propagation_speed, "solve_scattering")
    excitation isa AbstractExcitation &&
        _validate_scattering_excitation_wavenumber(
            excitation, k, "solve_scattering")

    # ── Step 1: Mesh validation ──
    mesh_report = mesh_resolution_report(mesh, frequency;
                                          points_per_wavelength=Float64(points_per_wavelength),
                                          c0=propagation_speed)

    if check_resolution && !mesh_report.meets_target
        msg = "Mesh under-resolved: edge_max/lambda=" *
              "$(round(mesh_report.edge_max_over_lambda, digits=3)), " *
              "target <= $(round(1.0/points_per_wavelength, digits=3)). " *
              "Refine the mesh, lower the frequency, or disable the check " *
              "only for a deliberate coarse-mesh study."
        push!(warnings, msg)
        if error_on_underresolved
            error("solve_scattering: $msg")
        elseif verbose
            println("  WARNING: $msg")
        end
    end

    # ── Step 2: Build RWG ──
    rwg = build_rwg(mesh)
    N = rwg.nedges
    N >= 1 ||
        throw(ArgumentError(
            "solve_scattering: mesh produces no RWG unknowns; at least two " *
            "triangles sharing an edge are required"))
    verbose && println(
        "  N = $N RWG unknowns " *
        "($(round(estimate_dense_matrix_gib(N), sigdigits=3)) GiB " *
        "one-matrix payload)")

    # ── Step 3: Method selection ──
    selected_method = method
    if method == :auto
        if N <= dense_direct_limit
            selected_method = :dense_direct
        elseif N <= dense_gmres_limit
            selected_method = :dense_gmres
        elseif N <= mlfma_threshold
            selected_method = :aca_gmres
        else
            selected_method = :mlfma
        end
        verbose && println("  Auto-selected method: $selected_method (N=$N)")
    else
        verbose && println("  Method: $selected_method (user-specified)")
    end
    if preconditioner == :ilu &&
       selected_method in (:dense_gmres, :aca_gmres)
        throw(ArgumentError(
            "solve_scattering: preconditioner=:ilu is supported only by the MLFMA path; " *
            "use :lu, :diag, :none, or :auto with $selected_method"))
    end

    if selected_method in (:dense_direct, :dense_gmres)
        dense_bytes = selected_method == :dense_direct ?
            _workflow_dense_direct_work_bytes(N) :
            _checked_array_payload_bytes(
                ComplexF64, N, N;
                label="solve_scattering dense EFIE matrix")
        _enforce_payload_limit(
            dense_bytes, max_dense_matrix_bytes,
            selected_method == :dense_direct ?
                "solve_scattering dense direct workspace" :
                "solve_scattering dense EFIE matrix",
            "max_dense_matrix_bytes")
    end

    # ── Step 4: Excitation vector ──
    local v::Vector{ComplexF64}
    if excitation isa AbstractVector
        length(excitation) == N ||
            throw(DimensionMismatch(
                "solve_scattering: excitation length " *
                "$(length(excitation)) != N=$N"))
        v = Vector{ComplexF64}(excitation)
    else
        v = assemble_excitation(mesh, rwg, excitation; quad_order=quad_order)
    end
    length(v) == N ||
        throw(DimensionMismatch(
            "solve_scattering: excitation length $(length(v)) != N=$N"))
    all(isfinite, v) ||
        throw(ArgumentError(
            "solve_scattering: excitation vector must contain only finite values"))

    # ── Step 5: Assembly ──
    local A_mlfma
    t_assembly = @elapsed begin
        if selected_method == :dense_direct || selected_method == :dense_gmres
            Z = assemble_Z_efie(
                mesh, rwg, k;
                quad_order=quad_order,
                mesh_precheck=false,
                max_output_bytes=max_dense_matrix_bytes)
        elseif selected_method == :aca_gmres
            A_aca = build_aca_operator(mesh, rwg, k;
                                       leaf_size=aca_leaf_size, eta=aca_eta,
                                       aca_tol=aca_tol, max_rank=aca_max_rank,
                                       quad_order=quad_order, mesh_precheck=false)
        elseif selected_method == :mlfma
            A_mlfma = build_mlfma_operator(mesh, rwg, k;
                                            quad_order=quad_order, verbose=verbose)
        end
    end
    verbose && println("  Assembly: $(round(t_assembly, digits=3)) s")

    if selected_method == :aca_gmres && verbose
        n_dense = length(A_aca.dense_blocks)
        n_lr = length(A_aca.lowrank_blocks)
        println("  ACA: $n_dense dense blocks, $n_lr low-rank blocks")
    end

    # ── Step 6: Preconditioner ──
    local P_nf
    t_precond = 0.0
    precond_used = preconditioner
    if selected_method == :dense_direct
        P_nf = nothing
        t_precond = 0.0
    elseif selected_method == :mlfma
        # MLFMA uses its built-in near-field matrix for preconditioning
        if preconditioner == :auto
            precond_used = :ilu
        end
        if precond_used == :none
            P_nf = nothing
        else
            factorization = precond_used == :diag ? :diag : (precond_used == :ilu ? :ilu : :lu)
            t_precond = @elapsed begin
                P_nf = build_nearfield_preconditioner(A_mlfma.Z_near;
                                                       factorization=factorization)
            end
            nnz_ratio = nnz(A_mlfma.Z_near) / N^2
            verbose && println("  Preconditioner ($precond_used): $(round(t_precond, digits=3)) s, " *
                               "nnz=$(round(nnz_ratio*100, digits=1))%")
        end
    else
        if preconditioner == :auto
            precond_used = :lu
        end

        if precond_used == :none
            P_nf = nothing
        else
            cutoff = nf_cutoff_lambda * lambda
            factorization = precond_used == :diag ? :diag : :lu
            t_precond = @elapsed begin
                if selected_method == :dense_gmres
                    P_nf = build_nearfield_preconditioner(Z, mesh, rwg, cutoff;
                                                           factorization=factorization)
                elseif selected_method == :aca_gmres
                    P_nf = build_nearfield_preconditioner(A_aca;
                                                           factorization=factorization)
                end
            end
            if verbose
                detail = selected_method == :dense_gmres ?
                    "cutoff=$(round(cutoff, sigdigits=3)) m " *
                    "(nf_cutoff_lambda=$(nf_cutoff_lambda))" :
                    "source=ACA inadmissible blocks"
                println(
                    "  Preconditioner ($precond_used): " *
                    "$(round(t_precond, digits=3)) s, $detail, " *
                    "nnz=$(round(P_nf.nnz_ratio * 100, digits=1))%")
            end
        end
    end

    # ── Step 7: Solve ──
    gmres_iters = -1
    gmres_residual = NaN
    local I_coeffs::Vector{ComplexF64}

    t_solve = @elapsed begin
        if selected_method == :dense_direct
            exact_work_bytes =
                _workflow_exact_dense_direct_work_bytes(N)
            enforce_exact_work = () -> _enforce_payload_limit(
                exact_work_bytes,
                max_dense_matrix_bytes,
                "solve_scattering exact dense direct workspace",
                "max_dense_matrix_bytes",
            )
            factor = _factor_dense_linear_system(
                Z,
                ComplexF64,
                "solve_scattering direct factorization";
                exact_fallback_check=enforce_exact_work,
            )
            I_coeffs = _solve_factored_linear_system(
                factor,
                Z,
                v,
                "solve_scattering direct solution";
                exact_fallback_check=enforce_exact_work,
            )
        elseif selected_method == :dense_gmres
            I_coeffs, stats = solve_gmres(Z, v;
                                           preconditioner=P_nf,
                                           tol=gmres_tol, maxiter=gmres_maxiter,
                                           check_gmres_convergence=check_gmres_convergence,
                                           check_true_residual=false)
            gmres_iters = stats.niter
        elseif selected_method == :aca_gmres
            I_coeffs, stats = solve_gmres(A_aca, v;
                                           preconditioner=P_nf,
                                           tol=gmres_tol, maxiter=gmres_maxiter,
                                           check_gmres_convergence=check_gmres_convergence,
                                           check_true_residual=false)
            gmres_iters = stats.niter
        elseif selected_method == :mlfma
            I_coeffs, stats = solve_gmres(A_mlfma, v;
                                           preconditioner=P_nf,
                                           tol=gmres_tol, maxiter=gmres_maxiter,
                                           check_gmres_convergence=check_gmres_convergence,
                                           check_true_residual=false)
            gmres_iters = stats.niter
        end
    end

    if selected_method != :dense_direct
        selected_operator = if selected_method == :dense_gmres
            Z
        elseif selected_method == :aca_gmres
            A_aca
        else
            A_mlfma
        end
        gmres_residual = if check_true_residual
            _assert_true_residual(
                selected_operator,
                I_coeffs,
                v,
                "solve_scattering";
                tol=gmres_tol,
                factor=true_residual_factor,
            )
        else
            _true_residual_ratio(
                selected_operator, I_coeffs, v, "solve_scattering")
        end
    end
    verbose && println("  Solve: $(round(t_solve, digits=3)) s" *
                       (gmres_iters >= 0 ? " ($gmres_iters GMRES iters)" : " (direct LU)"))

    return ScatteringResult(
        I_coeffs,
        selected_method,
        N,
        t_assembly,
        t_solve,
        t_precond,
        gmres_iters,
        gmres_residual,
        mesh_report,
        warnings,
    )
end
