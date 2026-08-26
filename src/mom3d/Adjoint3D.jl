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

function _validate_scalar_epsr_gradient_result_3d(
        res::DDAResult3D, nvoxels::Int)
    @inbounds for voxel in 1:nvoxels
        res.eps_r[voxel] isa Number ||
            throw(ArgumentError(
                "gradient_epsr_dda_3d supports only scalar-permittivity " *
                "DDA results; eps_r[$voxel] is a material tensor."))
        res.alpha[voxel] isa Number ||
            throw(ArgumentError(
                "gradient_epsr_dda_3d supports only scalar-polarizability " *
                "DDA results; alpha[$voxel] is a tensor."))
    end
    return nothing
end

@inline function _dda_material_gradient_cancellation_requires_exact(
        total::Float64, magnitude_sum::Float64, product_count::Int)
    iszero(magnitude_sum) && return false
    isfinite(total) && isfinite(magnitude_sum) || return true
    # Each primitive real product and each reduction contributes at most a
    # small multiple of eps to the forward error.  If the surviving value is
    # inside that bound, recompute from the stored Float64 factors exactly so
    # a finite cancellation is not silently returned with the wrong scale.
    error_factor = min(
        1.0,
        8eps(Float64) * max(product_count, 1),
    )
    return abs(total) <= error_factor * magnitude_sum
end

function _dda_material_gradient_fast(
        res::DDAResult3D,
        lambda_flat::Vector{ComplexF64},
        voxel::Int,
        dalpha::ComplexF64,
)
    total = 0.0
    magnitude_sum = 0.0
    product_count = 0
    source_center = res.grid.centers[voxel]
    source_field = res.E_total[voxel]
    @inbounds for observation_voxel in 1:res.grid.nvoxels
        observation_voxel == voxel && continue
        lambda_value = _read_field_component(
            lambda_flat, observation_voxel)
        interaction = _electric_dipole_alpha_apply_3d(
            res.grid.centers[observation_voxel],
            source_center,
            res.k0,
            dalpha,
            source_field,
        )
        for component in 1:3
            lambda_component = lambda_value[component]
            interaction_component = interaction[component]
            if _ieee_dense_extreme_factor(lambda_component, Float64) ||
               _ieee_dense_extreme_factor(interaction_component, Float64)
                return total, true
            end
            first_product =
                real(lambda_component) * real(interaction_component)
            second_product =
                imag(lambda_component) * imag(interaction_component)
            component_total = first_product + second_product
            if !isfinite(component_total) ||
               _scaled_component_sum_requires_exact(
                   first_product, second_product, component_total)
                return total, true
            end
            total += component_total
            magnitude_sum += abs(first_product) + abs(second_product)
            product_count += 2
            if !isfinite(total) || !isfinite(magnitude_sum)
                return total, true
            end
        end
    end
    return total,
           _dda_material_gradient_cancellation_requires_exact(
               total, magnitude_sum, product_count)
end

@noinline function _dda_material_gradient_bigfloat(
    res::DDAResult3D,
    lambda_flat::Vector{ComplexF64},
    voxel::Int,
    dalpha::ComplexF64,
)
    return setprecision(
            BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        total = zero(BigFloat)
        source_center = res.grid.centers[voxel]
        source_field = res.E_total[voxel]
        @inbounds for observation_voxel in 1:res.grid.nvoxels
            observation_voxel == voxel && continue
            lambda_value = _read_field_component(
                lambda_flat, observation_voxel)
            interaction = _electric_dipole_alpha_apply_3d(
                res.grid.centers[observation_voxel],
                source_center,
                res.k0,
                dalpha,
                source_field,
            )
            for component in 1:3
                lambda_component = lambda_value[component]
                interaction_component = interaction[component]
                total +=
                    BigFloat(real(lambda_component)) *
                    BigFloat(real(interaction_component)) +
                    BigFloat(imag(lambda_component)) *
                    BigFloat(imag(interaction_component))
            end
        end
        converted = Float64(2 * total)
        isfinite(converted) ||
            throw(OverflowError(
                "DDA material gradient at voxel $voxel is outside the " *
                "representable Float64 range."))
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
    solver in (:direct, :gmres) ||
        throw(ArgumentError(
            "Unsupported DDA adjoint solver: $solver " *
            "(expected :direct or :gmres)."))
    _validate_dda_result_3d(res; require_system=true)
    if solver == :direct && res.A_LU === nothing && !(res.A isa Matrix)
        throw(ArgumentError(
            "Direct DDA adjoint solving requires a stored dense forward " *
            "matrix or factorization; use solver=:gmres for a matrix-free " *
            "forward result."))
    end
    rhs = _coerce_adjoint_rhs_3d(grad_E_flat, res.grid.nvoxels, "grad_E_flat")

    if solver == :direct
        # Reuse the stored verified factorization of A when available. Solving
        # from existing factors is O(N^2), while refactorizing adjoint(res.A)
        # is O(N^3) per call (costly in optimization loops).
        adjoint_A = adjoint(res.A)
        adjoint_factorization = res.A_LU === nothing ?
                                _factor_dense_linear_system(
                                    adjoint_A,
                                    ComplexF64,
                                    "direct DDA adjoint factorization",
                                ) : adjoint(res.A_LU)
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
            check_true_residual=check_gmres_convergence,
        )
        return Vector{ComplexF64}(lambda)
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
    N = _validate_dda_result_3d(res)
    _validate_scalar_epsr_gradient_result_3d(res, N)
    res.radiative_correction &&
        throw(ArgumentError(
            "gradient_epsr_dda_3d requires an uncorrected " *
            "Clausius-Mossotti result; got radiative_correction=true. " *
            "Re-run solve_dda_3d with radiative_correction=false."))

    lambda_flat = _coerce_adjoint_rhs_3d(lambda, N, "lambda")
    grad = zeros(Float64, N)

    for j in 1:N
        Ej = res.E_total[j]
        dalpha = _dalpha_depsr_clausius_mossotti(
            ComplexF64(res.eps_r[j]), res.grid.volumes[j])
        total, needs_exact = _dda_material_gradient_fast(
            res, lambda_flat, j, dalpha)
        value = 2 * total
        if needs_exact || !isfinite(value)
            value = _dda_material_gradient_bigfloat(
                res, lambda_flat, j, dalpha)
        end
        grad[j] = value
    end

    return grad
end
