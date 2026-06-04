# Workflow.jl — High-level scattering solve with automatic method selection
#
# Provides `solve_scattering` which validates mesh resolution, selects the
# appropriate solver method (dense direct / dense GMRES / ACA GMRES / MLFMA)
# based on problem size, and handles preconditioner setup automatically.

export solve_scattering

const C0_DEFAULT = 299792458.0

"""
    solve_scattering(mesh, freq_hz, excitation; kwargs...)

High-level scattering solve that automatically selects the appropriate method
based on problem size:

- **N <= dense_direct_limit** (default 2000): Dense EFIE assembly + LU direct solve
- **dense_direct_limit < N <= dense_gmres_limit** (default 10000): Dense + NF-preconditioned GMRES
- **dense_gmres_limit < N <= mlfma_threshold** (default 50000): ACA H-matrix + NF-preconditioned GMRES
- **N > mlfma_threshold**: MLFMA O(N log N) + NF-preconditioned GMRES

Mesh resolution is validated against the frequency. Under-resolved meshes
produce a warning (or error if `error_on_underresolved=true`).

# Arguments
- `mesh::TriMesh`: triangle surface mesh
- `freq_hz::Real`: frequency in Hz
- `excitation`: either an `AbstractExcitation` or a pre-assembled `Vector{ComplexF64}` excitation vector

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

## NF preconditioner
- `nf_cutoff_lambda=1.0`: near-field cutoff in wavelengths
- `preconditioner=:auto`: one of `:auto`, `:lu`, `:diag`, `:none`

## ACA settings
- `aca_tol=1e-6`: ACA low-rank approximation tolerance
- `aca_leaf_size=64`: cluster tree leaf size
- `aca_eta=1.5`: admissibility parameter
- `aca_max_rank=50`: maximum rank per low-rank block

## General
- `verbose=true`: print progress info
- `quad_order=3`: quadrature order for EFIE entries
- `c0=299792458.0`: speed of light (m/s)

# Returns
A `ScatteringResult` with fields: `I_coeffs`, `method`, `N`, timing info,
GMRES stats, `mesh_report`, and `warnings`.
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
                          nf_cutoff_lambda::Float64=1.0,
                          preconditioner::Symbol=:auto,
                          aca_tol::Float64=1e-6,
                          aca_leaf_size::Int=64,
                          aca_eta::Float64=1.5,
                          aca_max_rank::Int=50,
                          verbose::Bool=true,
                          quad_order::Int=3,
                          c0::Real=C0_DEFAULT)
    freq_hz > 0 || error("solve_scattering: freq_hz must be > 0")
    method in (:auto, :dense_direct, :dense_gmres, :aca_gmres, :mlfma) ||
        error("solve_scattering: method must be :auto, :dense_direct, :dense_gmres, :aca_gmres, or :mlfma")

    warnings = String[]
    lambda = Float64(c0) / Float64(freq_hz)
    k = 2π * Float64(freq_hz) / Float64(c0)

    # ── Step 1: Mesh validation ──
    mesh_report = mesh_resolution_report(mesh, Float64(freq_hz);
                                          points_per_wavelength=Float64(points_per_wavelength),
                                          c0=Float64(c0))

    if check_resolution && !mesh_report.meets_target
        msg = "Mesh under-resolved: edge_max/lambda=$(round(mesh_report.edge_max_over_lambda, digits=3)), " *
              "target <= $(round(1.0/points_per_wavelength, digits=3)). " *
              "Results may be inaccurate."
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
    verbose && println("  N = $N RWG unknowns ($(round(estimate_dense_matrix_gib(N), sigdigits=3)) GiB dense)")

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

    # ── Step 4: Excitation vector ──
    local v::Vector{ComplexF64}
    if excitation isa AbstractVector
        v = Vector{ComplexF64}(excitation)
    else
        v = assemble_excitation(mesh, rwg, excitation; quad_order=quad_order)
    end
    length(v) == N || error("solve_scattering: excitation length $(length(v)) != N=$N")

    # ── Step 5: Assembly ──
    local A_mlfma
    t_assembly = @elapsed begin
        if selected_method == :dense_direct || selected_method == :dense_gmres
            Z = assemble_Z_efie(mesh, rwg, k; quad_order=quad_order, mesh_precheck=false)
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
            verbose && println("  Preconditioner ($precond_used): $(round(t_precond, digits=3)) s, " *
                               "cutoff=$(round(cutoff, sigdigits=3)) m ($(nf_cutoff_lambda)lambda), " *
                               "nnz=$(round(P_nf.nnz_ratio*100, digits=1))%")
        end
    end

    # ── Step 7: Solve ──
    gmres_iters = -1
    gmres_residual = NaN
    local I_coeffs::Vector{ComplexF64}

    t_solve = @elapsed begin
        if selected_method == :dense_direct
            I_coeffs = Z \ v
        elseif selected_method == :dense_gmres
            I_coeffs, stats = solve_gmres(Z, v;
                                           preconditioner=P_nf,
                                           tol=gmres_tol, maxiter=gmres_maxiter)
            gmres_iters = stats.niter
            gmres_residual = isempty(stats.residuals) ? NaN : stats.residuals[end]
        elseif selected_method == :aca_gmres
            I_coeffs, stats = solve_gmres(A_aca, v;
                                           preconditioner=P_nf,
                                           tol=gmres_tol, maxiter=gmres_maxiter)
            gmres_iters = stats.niter
            gmres_residual = isempty(stats.residuals) ? NaN : stats.residuals[end]
        elseif selected_method == :mlfma
            I_coeffs, stats = solve_gmres(A_mlfma, v;
                                           preconditioner=P_nf,
                                           tol=gmres_tol, maxiter=gmres_maxiter)
            gmres_iters = stats.niter
            gmres_residual = isempty(stats.residuals) ? NaN : stats.residuals[end]
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
