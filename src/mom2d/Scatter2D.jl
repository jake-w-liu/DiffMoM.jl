# Scatter2D.jl — Scattered field computation and Jacobian for 2D VIE
#
# E_scat(r_obs) = k₀² Σ_n χ_n E_n G₂D(r_obs, r_n) A_n
#
# Jacobian: ∂E_scat/∂χ computed via implicit differentiation of the VIE system

export scattered_field_2d, green_obs_matrix, jacobian_scattered_field_2d

"""
    green_obs_matrix(r_obs, mesh, k0;
                     max_output_bytes=2_000_000_000)

Compute the observation Green's function matrix G_obs[m,n] = G₂D(r_obs[m], r_n).
Observation points must be outside the scattering domain.
"""
function green_obs_matrix(
        r_obs::AbstractVector{Vec2}, mesh::Mesh2D, k0::Float64;
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validate_mesh_2d(mesh)
    _validate_positive_finite_2d(k0, "green_obs_matrix wavenumber")
    _validate_observation_points_2d(r_obs, mesh, "green_obs_matrix")
    payload_bytes = _checked_array_payload_bytes(
        ComplexF64, length(r_obs), mesh.ncells;
        label="green_obs_matrix output")
    _enforce_payload_limit(
        payload_bytes, max_output_bytes,
        "green_obs_matrix output", "max_output_bytes")
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
    scattered_field_2d(vie_result, r_obs; max_output_bytes=2_000_000_000)

Compute scattered field at observation points using solved VIE result.
E_scat(r_obs) = k₀² Σ_n χ_n E_n G₂D(r_obs, r_n) A_n

`max_output_bytes` caps the raw payload of the returned vector before any
observation-point work is performed.
"""
function scattered_field_2d(
        vr::VIEResult2D, r_obs::AbstractVector{Vec2};
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    M = length(r_obs)
    output_bytes = _checked_array_payload_bytes(
        ComplexF64, M; label="scattered_field_2d output")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "scattered_field_2d output", "max_output_bytes")
    _validate_vie_result_2d(vr)
    _validate_observation_points_2d(
        r_obs, vr.mesh, "scattered_field_2d")
    k0sq = vr.k0^2
    stored_square_usable = isfinite(k0sq) && k0sq >= floatmin(Float64)
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
    jacobian_scattered_field_2d(
        vie_result, r_obs; max_work_bytes=2_000_000_000)

Compute the Jacobian J[m,p] = ∂E_scat(r_obs[m])/∂χ_p.

Uses implicit differentiation: Z E = E^inc ⟹ ∂E/∂χ_p = k₀² E_p Z⁻¹ D[:,p].
For reciprocal D, evaluates the equivalent factor
W = (I - k₀² diag(χ)D)⁻¹ = Z⁻ᵀ through the cached LU factorization.

Returns (J, G_obs) where J is M_rx × N_cells.
"""
function jacobian_scattered_field_2d(
        vr::VIEResult2D, r_obs::AbstractVector{Vec2};
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validate_vie_result_2d(vr; require_system=true)
    _validate_observation_points_2d(
        r_obs, vr.mesh, "jacobian_scattered_field_2d")
    rectangular_bytes = _checked_array_payload_bytes(
        ComplexF64, length(r_obs), vr.mesh.ncells;
        label="jacobian_scattered_field_2d rectangular matrix")
    # G_obs, the transposed sensitivity solve, and the returned Jacobian have
    # identical raw payloads and coexist on the ordinary path.
    work_bytes = try
        Base.Checked.checked_mul(3, rectangular_bytes)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "jacobian_scattered_field_2d raw-work estimate overflows Int"))
    end
    _enforce_payload_limit(
        work_bytes, max_work_bytes,
        "jacobian_scattered_field_2d dense matrices", "max_work_bytes")
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
    # W = (I - k₀² diag(χ)D)⁻¹ = Z⁻ᵀ. Evaluate G_obs*W through
    # (G_obs*Z⁻ᵀ)ᵀ = Z⁻¹*G_obsᵀ. This reuses the cached forward LU and needs
    # an N×M workspace instead of materializing the N×N inverse.
    sensitivity_transpose = Matrix{ComplexF64}(undef, N, M)
    sensitivity_transpose = _solve_factored_linear_system!(
        sensitivity_transpose, vr.Z_LU, vr.Z,
        transpose(G_obs), "jacobian_scattered_field_2d")

    factor_backend = _direct_factorization_backend(vr.Z_LU)
    if factor_backend isa _BigFloatDenseLUPlan
        # In the exact-factor branch, converting Z⁻¹Gᵀ to ComplexF64 before
        # multiplying by k₀² can erase a representable final Jacobian entry.
        # Retain the exact solve through the complete physical product.
        J = setprecision(BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
            big_type = Complex{BigFloat}
            rhs_big = Matrix{big_type}(undef, N, M)
            @inbounds for p in 1:N, m in 1:M
                rhs_big[p, m] = big_type(G_obs[m, p])
            end
            sensitivity_big = factor_backend.factorization \ rhs_big
            result = Matrix{ComplexF64}(undef, M, N)
            scale = BigFloat(vr.k0)^2 * BigFloat(A)
            @inbounds for p in 1:N, m in 1:M
                value = scale * big_type(vr.E_total[p]) *
                        sensitivity_big[p, m]
                result[m, p] = ComplexF64(value)
                isfinite(result[m, p]) ||
                    throw(OverflowError(
                        "jacobian_scattered_field_2d entry is outside " *
                        "the representable ComplexF64 range."))
            end
            result
        end
        return J, G_obs
    end

    # J = k₀² A × (G_obs × W) × diag(E).
    J = Matrix{ComplexF64}(undef, M, N)
    @inbounds for p in 1:N
        for m in 1:M
            J[m, p] = if stored_square_usable
                _range_safe_product_2d(
                    k0sq, A, vr.E_total[p],
                    sensitivity_transpose[p, m],
                    "jacobian_scattered_field_2d entry")
            else
                _range_safe_product_2d(
                    vr.k0, vr.k0, vr.E_total[p],
                    sensitivity_transpose[p, m], A,
                    "jacobian_scattered_field_2d entry")
            end
        end
    end

    all(isfinite, J) ||
        error("jacobian_scattered_field_2d produced non-finite values.")
    return J, G_obs
end
