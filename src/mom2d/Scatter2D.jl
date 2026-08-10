# Scatter2D.jl — Scattered field computation and Jacobian for 2D VIE
#
# E_scat(r_obs) = k₀² Σ_n χ_n E_n G₂D(r_obs, r_n) A_n
#
# Jacobian: ∂E_scat/∂χ computed via implicit differentiation of the VIE system

export scattered_field_2d, green_obs_matrix, jacobian_scattered_field_2d

"""
    green_obs_matrix(r_obs, mesh, k0)

Compute the observation Green's function matrix G_obs[m,n] = G₂D(r_obs[m], r_n).
Observation points must be outside the scattering domain.
"""
function green_obs_matrix(r_obs::AbstractVector{Vec2}, mesh::Mesh2D, k0::Float64)
    _validate_mesh_2d(mesh)
    _validate_positive_finite_2d(k0, "green_obs_matrix wavenumber")
    _validate_observation_points_2d(r_obs, mesh, "green_obs_matrix")
    return _green_obs_matrix_unchecked(r_obs, mesh, k0)
end

function _validate_observation_points_2d(
    r_obs::AbstractVector{Vec2},
    mesh::Mesh2D,
    label::AbstractString,
)
    x1 = muladd(mesh.nx, mesh.dx, mesh.x0)
    y1 = muladd(mesh.ny, mesh.dy, mesh.y0)
    isfinite(x1) && isfinite(y1) ||
        throw(ArgumentError("$label mesh bounds must be finite."))
    @inbounds for m in eachindex(r_obs)
        point = r_obs[m]
        all(isfinite, point) ||
            throw(ArgumentError(
                "$label observation point $m must be finite, got $point."))
        inside_x = mesh.x0 <= point[1] <= x1
        inside_y = mesh.y0 <= point[2] <= y1
        if inside_x && inside_y
            throw(DomainError(
                point,
                "$label observation point $m must lie outside the Mesh2D domain.",
            ))
        end
    end
    return nothing
end

function _green_obs_matrix_unchecked(
    r_obs::AbstractVector{Vec2},
    mesh::Mesh2D,
    k0::Float64,
)
    M = length(r_obs)
    N = mesh.ncells
    G_obs = Matrix{ComplexF64}(undef, M, N)
    @inbounds for n in 1:N
        for m in 1:M
            G_obs[m, n] =
                _greens_2d_unchecked(r_obs[m], mesh.centers[n], k0)
        end
    end
    all(isfinite, G_obs) ||
        error("green_obs_matrix produced non-finite entries.")
    return G_obs
end

@noinline function _scattered_field_sum_big_2d(
        vr::VIEResult2D, observation::Vec2)
    return setprecision(BigFloat, _VIE_PRODUCT_FALLBACK_PRECISION_2D) do
        k_big = BigFloat(vr.k0)
        area_big = BigFloat(vr.mesh.cell_area)
        field = zero(Complex{BigFloat})
        @inbounds for n in 1:vr.mesh.ncells
            (iszero(vr.chi[n]) || iszero(vr.E_total[n])) && continue
            green = _greens_2d_unchecked(
                observation, vr.mesh.centers[n], vr.k0)
            field += k_big^2 * BigFloat(vr.chi[n]) *
                     Complex{BigFloat}(vr.E_total[n]) *
                     Complex{BigFloat}(green) * area_big
        end
        converted = ComplexF64(field)
        isfinite(converted) ||
            throw(OverflowError(
                "scattered_field_2d result is outside the representable ComplexF64 range."))
        return converted
    end
end

function _scattered_field_sum_2d(
        vr::VIEResult2D,
        observation::Vec2,
        k0sq::Float64,
        stored_square_usable::Bool)
    field = zero(ComplexF64)
    try
        @inbounds for n in 1:vr.mesh.ncells
            green = _greens_2d_unchecked(
                observation, vr.mesh.centers[n], vr.k0)
            contribution = if stored_square_usable
                _range_safe_product_2d(
                    k0sq, vr.chi[n], vr.E_total[n], green,
                    vr.mesh.cell_area,
                    "scattered_field_2d source contribution")
            else
                _range_safe_product_2d(
                    vr.k0, vr.k0, vr.chi[n], vr.E_total[n], green,
                    vr.mesh.cell_area,
                    "scattered_field_2d source contribution")
            end
            next_field = field + contribution
            isfinite(next_field) ||
                return _scattered_field_sum_big_2d(vr, observation)
            field = next_field
        end
    catch err
        err isa OverflowError || rethrow()
        return _scattered_field_sum_big_2d(vr, observation)
    end
    return field
end

"""
    scattered_field_2d(vie_result, r_obs)

Compute scattered field at observation points using solved VIE result.
E_scat(r_obs) = k₀² Σ_n χ_n E_n G₂D(r_obs, r_n) A_n
"""
function scattered_field_2d(vr::VIEResult2D, r_obs::AbstractVector{Vec2})
    _validate_vie_result_2d(vr)
    _validate_observation_points_2d(
        r_obs, vr.mesh, "scattered_field_2d")
    k0sq = vr.k0^2
    stored_square_usable = isfinite(k0sq) && k0sq >= floatmin(Float64)
    M = length(r_obs)

    E_scat = zeros(ComplexF64, M)
    @inbounds for m in 1:M
        E_scat[m] = _scattered_field_sum_2d(
            vr, r_obs[m], k0sq, stored_square_usable)
    end
    all(isfinite, E_scat) ||
        error("scattered_field_2d produced non-finite field values.")
    return E_scat
end

"""
    jacobian_scattered_field_2d(vie_result, r_obs)

Compute the Jacobian J[m,p] = ∂E_scat(r_obs[m])/∂χ_p.

Uses implicit differentiation: Z E = E^inc ⟹ ∂E/∂χ_p = k₀² E_p Z⁻¹ D[:,p].
For reciprocal D, evaluates the equivalent factor
W = (I - k₀² diag(χ)D)⁻¹ = Z⁻ᵀ through the cached LU factorization.

Returns (J, G_obs) where J is M_rx × N_cells.
"""
function jacobian_scattered_field_2d(vr::VIEResult2D, r_obs::AbstractVector{Vec2})
    _validate_vie_result_2d(vr; require_system=true)
    _validate_observation_points_2d(
        r_obs, vr.mesh, "jacobian_scattered_field_2d")
    G_obs = _green_obs_matrix_unchecked(r_obs, vr.mesh, vr.k0)
    A = vr.mesh.cell_area
    k0sq = vr.k0^2
    stored_square_usable = isfinite(k0sq) && k0sq >= floatmin(Float64)
    N = vr.mesh.ncells
    M = length(r_obs)

    # The direct sensitivity expression contains
    # W = I + k₀² diag(χ) Z⁻¹D. Forming it that way can underflow Z⁻¹D
    # before the product is rescaled, or lose W through cancellation. Since
    # D is reciprocal (Dᵀ = D), the push-through identity gives
    # W = (I - k₀² diag(χ)D)⁻¹ = Z⁻ᵀ. Solving the transposed system directly
    # preserves the available exponent range and needs only one N×N workspace.
    W = Matrix{ComplexF64}(I, N, N)
    ldiv!(transpose(vr.Z_LU), W)

    # J = k₀² A × G_obs × W × diag(E). Multiply directly into J rather
    # than materializing a separate G_obs*W matrix.
    J = Matrix{ComplexF64}(undef, M, N)
    mul!(J, G_obs, W)
    @inbounds for p in 1:N
        for m in 1:M
            J[m, p] = if stored_square_usable
                _range_safe_product_2d(
                    k0sq, A, vr.E_total[p], J[m, p],
                    "jacobian_scattered_field_2d entry")
            else
                _range_safe_product_2d(
                    vr.k0, vr.k0, vr.E_total[p], J[m, p], A,
                    "jacobian_scattered_field_2d entry")
            end
        end
    end

    all(isfinite, J) ||
        error("jacobian_scattered_field_2d produced non-finite values.")
    return J, G_obs
end
