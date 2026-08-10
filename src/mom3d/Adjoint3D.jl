# Adjoint3D.jl -- Material sensitivities for the 3D DDA path

export solve_dda_adjoint_3d, gradient_epsr_dda_3d

function _coerce_adjoint_rhs_3d(grad_E, n::Int, label::AbstractString)
    if length(grad_E) == n
        return _flatten_fields_3d(grad_E, n, label)
    elseif length(grad_E) == 3n
        return _copy_finite_complex_vector_3d(grad_E, 3n, label)
    else
        error("$label length ($(length(grad_E))) must be nvoxels ($n) or 3*nvoxels ($(3n)).")
    end
end

function _dalpha_depsr_clausius_mossotti(eps_r::ComplexF64, volume::Real)
    abs(eps_r + 2) > 100 * eps(Float64) ||
        error("Clausius-Mossotti polarizability derivative is singular for eps_r near -2.")
    V = Float64(volume)
    derivative = 9 * V / (eps_r + 2)^2
    isfinite(derivative) && !iszero(derivative) && return derivative

    return setprecision(BigFloat, _POLARIZABILITY_FALLBACK_PRECISION) do
        eps_big = Complex{BigFloat}(eps_r)
        value = 9 * BigFloat(V) / (eps_big + 2)^2
        converted = ComplexF64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "Clausius-Mossotti polarizability derivative is outside " *
                "the representable ComplexF64 range."))
        return converted
    end
end

"""
    solve_dda_adjoint_3d(res, grad_E_flat; solver=:direct, tol=1e-8,
                         maxiter=200, memory=20,
                         check_gmres_convergence=true)

Solve the 3D DDA adjoint system

    A' * lambda = grad_E_flat

for an existing `DDAResult3D`. For `J = real(E' * Q * E)`, pass
`grad_E_flat = Q * E`; `gradient_epsr_dda_3d` applies the corresponding factor
of two for real design parameters.
"""
function solve_dda_adjoint_3d(res::DDAResult3D, grad_E_flat;
                              solver::Symbol=:direct,
                              tol::Float64=1e-8,
                              maxiter::Int=200,
                              memory::Int=20,
                              verbose::Bool=false,
                              check_gmres_convergence::Bool=true)
    rhs = _coerce_adjoint_rhs_3d(grad_E_flat, res.grid.nvoxels, "grad_E_flat")

    if solver == :direct
        # Reuse the stored LU factorization of A when available: solving A' x = rhs
        # from existing factors is O(N^2) and numerically identical to refactorizing
        # adjoint(res.A), which is O(N^3) per call (costly in optimization loops).
        adjoint_A = adjoint(res.A)
        adjoint_factorization = res.A_LU === nothing ?
                                lu(adjoint_A) : adjoint(res.A_LU)
        lambda = _solve_factored_linear_system(
            adjoint_factorization,
            adjoint_A,
            rhs,
            "direct DDA adjoint solution",
        )
        return Vector{ComplexF64}(_assert_finite_linear_vector(
            lambda, "direct DDA adjoint solution"))
    elseif solver == :gmres
        lambda, stats = solve_gmres_adjoint(
            res.A, rhs;
            memory=memory,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
            check_gmres_convergence=check_gmres_convergence,
        )
        return Vector{ComplexF64}(lambda)
    else
        error("Unsupported DDA adjoint solver: $solver (expected :direct or :gmres).")
    end
end

"""
    gradient_epsr_dda_3d(res, lambda)

Return the real gradient with respect to one real scalar `eps_r` design
parameter per voxel. This uses

    alpha = 3V * (eps_r - 1) / (eps_r + 2)
    d alpha / d eps_r = 9V / (eps_r + 2)^2

and the DDA system convention `A_ij = delta_ij - G_ij * alpha_j`.
"""
function gradient_epsr_dda_3d(res::DDAResult3D, lambda)
    res.radiative_correction &&
        error("gradient_epsr_dda_3d currently supports uncorrected Clausius-Mossotti alpha only.")

    N = res.grid.nvoxels
    lambda_flat = _coerce_adjoint_rhs_3d(lambda, N, "lambda")
    grad = zeros(Float64, N)

    for j in 1:N
        Ej = res.E_total[j]
        dalpha = _dalpha_depsr_clausius_mossotti(
            res.eps_r[j], res.grid.volumes[j])
        acc = 0.0 + 0im
        rj = res.grid.centers[j]
        for i in 1:N
            i == j && continue
            lambdai = _read_field_component(lambda_flat, i)
            GEj = _electric_dipole_alpha_apply_3d(
                res.grid.centers[i], rj, res.k0, dalpha, Ej)
            acc += dot(lambdai, GEj)
        end
        value = 2 * real(acc)
        isfinite(value) ||
            throw(OverflowError(
                "DDA material gradient at voxel $j is outside the " *
                "representable Float64 range."))
        grad[j] = value
    end

    return grad
end
