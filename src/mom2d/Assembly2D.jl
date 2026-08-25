# Assembly2D.jl — 2D TM Volume Integral Equation assembly and solve
#
# VIE for TM scattering from inhomogeneous dielectric:
#   E_z(r) = E_z^inc(r) + k₀² ∫_D χ(r') G₂D(r,r') E_z(r') dA'
#
# MoM discretization (pulse basis, point matching):
#   Z E = E^inc,  where Z = I - k₀² D diag(χ)   (χ_n scales column n: Z[m,n] = δ - k₀² χ_n D[m,n])

export assemble_vie_2d, solve_vie_2d

const _VIE_PRODUCT_FALLBACK_PRECISION_2D = 256
const _VIE_RHS_SCALE_LOWER_2D = sqrt(floatmin(Float64))
const _VIE_RHS_SCALE_UPPER_2D = sqrt(floatmax(Float64))
const _VIE_SOLVE_MIN_FALLBACK_PRECISION_2D = 256
const _VIE_SOLVE_FALLBACK_GUARD_BITS_2D = 128

@inline _complex_component_scale_2d(value::ComplexF64) =
    max(abs(real(value)), abs(imag(value)))

@inline function _usable_nonzero_product_2d(value::ComplexF64)
    isfinite(value) || return false
    scale = _complex_component_scale_2d(value)
    return scale >= floatmin(Float64)
end

@inline function _scaled_components_are_normal_2d(
        scaled::ComplexF64, unscaled::ComplexF64)
    real_ok = iszero(real(unscaled)) || abs(real(scaled)) >= floatmin(Float64)
    imag_ok = iszero(imag(unscaled)) || abs(imag(scaled)) >= floatmin(Float64)
    return real_ok && imag_ok
end

function _vie_reciprocal_condition_2d(factorization, Z::Matrix{ComplexF64})
    matrix_norm = opnorm(Z, 1)
    isfinite(matrix_norm) || return 0.0
    iszero(matrix_norm) && return 0.0
    reciprocal_condition = LinearAlgebra.LAPACK.gecon!(
        '1', factorization.factors, matrix_norm)
    isfinite(reciprocal_condition) || return 0.0
    return reciprocal_condition
end

@inline function _vie_solve_fallback_precision_2d(
        reciprocal_condition::Float64,
        label::AbstractString)
    reciprocal_condition > 0.0 ||
        error(
            "$label system is too ill-conditioned for a reliable " *
            "Float64 reciprocal-condition estimate.")
    condition_bits = ceil(Int, -log2(reciprocal_condition))
    return max(
        _VIE_SOLVE_MIN_FALLBACK_PRECISION_2D,
        condition_bits + _VIE_SOLVE_FALLBACK_GUARD_BITS_2D)
end

@noinline function _solve_vie_high_precision_2d(
        Z::Matrix{ComplexF64},
        rhs::Union{AbstractVector{ComplexF64},AbstractMatrix{ComplexF64}},
        reciprocal_condition::Float64,
        label::AbstractString)
    precision_bits = _vie_solve_fallback_precision_2d(
        reciprocal_condition, label)
    return setprecision(BigFloat, precision_bits) do
        Z_big = Complex{BigFloat}.(Z)
        rhs_big = Complex{BigFloat}.(rhs)
        solution_big = ldiv!(lu(Z_big), rhs_big)
        solution = ComplexF64.(solution_big)
        all(isfinite, solution) ||
            error(
                "$label high-precision solution is outside the " *
                "representable ComplexF64 range.")
        return solution
    end
end

function _solve_vie_factored_2d!(
        workspace::Union{AbstractVector{ComplexF64},AbstractMatrix{ComplexF64}},
        factorization,
        Z::Matrix{ComplexF64},
        label::AbstractString,
        reciprocal_condition::Float64 =
            _vie_reciprocal_condition_2d(factorization, Z))
    reciprocal_condition <= eps(Float64) &&
        return _solve_vie_high_precision_2d(
            Z, workspace, reciprocal_condition, label)

    rhs_scale = 0.0
    @inbounds for value in workspace
        rhs_scale = max(rhs_scale, _complex_component_scale_2d(value))
    end

    if !iszero(rhs_scale) &&
       (rhs_scale < _VIE_RHS_SCALE_LOWER_2D ||
        rhs_scale > _VIE_RHS_SCALE_UPPER_2D)
        workspace ./= rhs_scale
        ldiv!(factorization, workspace)
        @inbounds for index in eachindex(workspace)
            workspace[index] *= rhs_scale
        end
        return workspace
    end
    ldiv!(factorization, workspace)
    return workspace
end

function _factor_vie_system_2d(Z::Matrix{ComplexF64}, label::AbstractString)
    raw_factor = try
        lu(Z)
    catch err
        _recoverable_direct_solve_error(err) || rethrow()
        nothing
    end
    if raw_factor !== nothing && issuccess(raw_factor)
        reciprocal_condition = _vie_reciprocal_condition_2d(raw_factor, Z)
        reciprocal_condition > eps(Float64) && return raw_factor
    end
    # A small componentwise residual does not imply an accurate field for a
    # severely ill-conditioned VIE system. Factor the stored Float64 matrix
    # exactly in the bounded high-precision backend and cache that plan for
    # both the forward solve and subsequent Jacobian block solves.
    return _factor_bigfloat_ieee_matrix(Z, ComplexF64, label)
end

@noinline function _product_bigfloat_2d(
        first::Float64,
        second::Float64,
        third::ComplexF64,
        label::AbstractString)
    return setprecision(BigFloat, _VIE_PRODUCT_FALLBACK_PRECISION_2D) do
        value = BigFloat(first) * BigFloat(second) *
                Complex{BigFloat}(third)
        converted = ComplexF64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "$label is outside the representable ComplexF64 range."))
        return converted
    end
end

@noinline function _product_bigfloat_2d(
        first::Float64,
        second::Float64,
        third::Float64,
        fourth::ComplexF64,
        fifth::ComplexF64,
        sixth::Float64,
        label::AbstractString)
    return setprecision(BigFloat, _VIE_PRODUCT_FALLBACK_PRECISION_2D) do
        value = BigFloat(first) * BigFloat(second) * BigFloat(third) *
                Complex{BigFloat}(fourth) * Complex{BigFloat}(fifth) *
                BigFloat(sixth)
        converted = ComplexF64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "$label is outside the representable ComplexF64 range."))
        return converted
    end
end

@noinline function _product_bigfloat_2d(
        first::Float64,
        second::Float64,
        third::ComplexF64,
        fourth::ComplexF64,
        label::AbstractString)
    return setprecision(BigFloat, _VIE_PRODUCT_FALLBACK_PRECISION_2D) do
        value = BigFloat(first) * BigFloat(second) *
                Complex{BigFloat}(third) * Complex{BigFloat}(fourth)
        converted = ComplexF64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "$label is outside the representable ComplexF64 range."))
        return converted
    end
end

@noinline function _product_bigfloat_2d(
        first::Float64,
        second::Float64,
        third::ComplexF64,
        fourth::ComplexF64,
        fifth::Float64,
        label::AbstractString)
    return setprecision(BigFloat, _VIE_PRODUCT_FALLBACK_PRECISION_2D) do
        value = BigFloat(first) * BigFloat(second) *
                Complex{BigFloat}(third) * Complex{BigFloat}(fourth) *
                BigFloat(fifth)
        converted = ComplexF64(value)
        isfinite(converted) ||
            throw(OverflowError(
                "$label is outside the representable ComplexF64 range."))
        return converted
    end
end

@inline function _range_safe_product_2d(
        first::Float64,
        second::Float64,
        third::ComplexF64,
        label::AbstractString)
    (iszero(first) || iszero(second) || iszero(third)) &&
        return zero(ComplexF64)

    value = (first * second) * third
    _usable_nonzero_product_2d(value) && return value
    value = (first * third) * second
    _usable_nonzero_product_2d(value) && return value
    value = (second * third) * first
    _usable_nonzero_product_2d(value) && return value
    return _product_bigfloat_2d(first, second, third, label)
end

@inline function _range_safe_product_2d(
        first::Float64,
        second::Float64,
        third::Float64,
        fourth::ComplexF64,
        fifth::ComplexF64,
        sixth::Float64,
        label::AbstractString)
    (iszero(first) || iszero(second) || iszero(third) ||
     iszero(fourth) || iszero(fifth) || iszero(sixth)) &&
        return zero(ComplexF64)

    value = (first * sixth) * (second * third) * (fourth * fifth)
    _usable_nonzero_product_2d(value) && return value
    value = (first * fourth) * (second * fifth) * (third * sixth)
    _usable_nonzero_product_2d(value) && return value
    value = (first * third) * (second * sixth) * (fourth * fifth)
    _usable_nonzero_product_2d(value) && return value
    value = (first * fifth) * (second * sixth) * (third * fourth)
    _usable_nonzero_product_2d(value) && return value
    value = (first * second) * (third * sixth) * (fourth * fifth)
    _usable_nonzero_product_2d(value) && return value
    return _product_bigfloat_2d(
        first, second, third, fourth, fifth, sixth, label)
end

@inline function _range_safe_product_2d(
        first::Float64,
        second::Float64,
        third::ComplexF64,
        fourth::ComplexF64,
        label::AbstractString)
    (iszero(first) || iszero(second) || iszero(third) || iszero(fourth)) &&
        return zero(ComplexF64)

    value = (first * second) * (third * fourth)
    _usable_nonzero_product_2d(value) && return value
    value = (first * third) * (second * fourth)
    _usable_nonzero_product_2d(value) && return value
    value = (first * fourth) * (second * third)
    _usable_nonzero_product_2d(value) && return value
    return _product_bigfloat_2d(
        first, second, third, fourth, label)
end

@inline function _range_safe_product_2d(
        first::Float64,
        second::Float64,
        third::ComplexF64,
        fourth::ComplexF64,
        fifth::Float64,
        label::AbstractString)
    (iszero(first) || iszero(second) || iszero(third) ||
     iszero(fourth) || iszero(fifth)) && return zero(ComplexF64)

    value = (first * fifth) * (second * third) * fourth
    _usable_nonzero_product_2d(value) && return value
    value = (first * fourth) * (second * third) * fifth
    _usable_nonzero_product_2d(value) && return value
    value = (first * second) * (third * fourth) * fifth
    _usable_nonzero_product_2d(value) && return value
    value = (first * third) * (second * fourth) * fifth
    _usable_nonzero_product_2d(value) && return value
    return _product_bigfloat_2d(
        first, second, third, fourth, fifth, label)
end

"""
    assemble_vie_2d(mesh, k0, chi;
                    max_output_bytes=2_000_000_000)

Assemble VIE system matrix: Z[m,n] = δ[m,n] - k₀² χ[n] D[m,n]

Returns (Z, D) where D is the Green's function integral matrix.
"""
function assemble_vie_2d(
        mesh::Mesh2D, k0::Float64, chi::AbstractVector{Float64};
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    _validate_mesh_2d(mesh)
    _validate_positive_finite_2d(k0, "assemble_vie_2d wavenumber")
    length(chi) == mesh.ncells ||
        throw(DimensionMismatch(
            "chi length $(length(chi)) must match $(mesh.ncells) mesh cells."))
    all(isfinite, chi) ||
        throw(ArgumentError("chi must contain only finite values."))
    N = mesh.ncells
    matrix_bytes = _checked_array_payload_bytes(
        ComplexF64, N, N; label="assemble_vie_2d matrix")
    output_bytes = try
        Base.Checked.checked_mul(2, matrix_bytes)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "assemble_vie_2d output-payload estimate overflows Int"))
    end
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "assemble_vie_2d Z and D matrices", "max_output_bytes")
    area = mesh.cell_area
    a_eq = _equivalent_radius_unchecked(mesh)
    D_self = self_cell_integral_2d(k0, a_eq)
    ka = k0 * a_eq
    k0sq = k0^2
    stored_square_usable = isfinite(k0sq) && k0sq >= floatmin(Float64)
    stored_self_usable = stored_square_usable &&
                         abs(real(D_self)) >= floatmin(Float64) &&
                         abs(imag(D_self)) >= floatmin(Float64)
    small_self = ka <= _SELF_CELL_SERIES_CUTOFF_2D
    self_kernel = if small_self
        _small_self_cell_normalized_2d(k0, a_eq, ka)
    else
        H1 = besselh(1, 2, ka)
        isfinite(H1) ||
            error("assemble_vie_2d produced a non-finite self-cell Hankel value.")
        value = (-im * π / 2) * (ka * H1 - 2im / π)
        isfinite(value) ||
            error("assemble_vie_2d produced a non-finite scaled self-cell value.")
        value
    end

    D = Matrix{ComplexF64}(undef, N, N)
    Z = Matrix{ComplexF64}(undef, N, N)

    @inbounds for n in 1:N
        chi_n = ComplexF64(chi[n])
        self_coefficient = if stored_self_usable
            _range_safe_product_2d(
                k0sq, chi[n], D_self,
                "assemble_vie_2d self-cell interaction coefficient")
        elseif small_self
            _range_safe_product_2d(
                ka, ka, chi_n, self_kernel,
                "assemble_vie_2d self-cell interaction coefficient")
        else
            _range_safe_product_2d(
                1.0, chi[n], self_kernel,
                "assemble_vie_2d self-cell interaction coefficient")
        end
        for m in 1:N
            if m == n
                D[m, n] = D_self
                Z[m, n] = -self_coefficient
            else
                green = _greens_2d_unchecked(
                    mesh.centers[m], mesh.centers[n], k0)
                D_entry = green * area
                D[m, n] = D_entry
                coefficient = if stored_square_usable &&
                                 _scaled_components_are_normal_2d(D_entry, green)
                    _range_safe_product_2d(
                        k0sq, chi[n], D_entry,
                        "assemble_vie_2d interaction coefficient")
                else
                    _range_safe_product_2d(
                        k0, k0, chi_n, green, area,
                        "assemble_vie_2d interaction coefficient")
                end
                Z[m, n] = -coefficient
            end
        end
        Z[n, n] += 1.0  # add identity
    end

    all(isfinite, D) ||
        error("assemble_vie_2d produced non-finite Green-matrix entries.")
    all(isfinite, Z) ||
        error("assemble_vie_2d produced non-finite system-matrix entries.")
    return Z, D
end

"""
    solve_vie_2d(mesh, k0, chi, E_inc;
                 max_output_bytes=2_000_000_000)

Solve the 2D VIE for internal total fields.
Returns `VIEResult2D` with all computed quantities for downstream use.
"""
function solve_vie_2d(
        mesh::Mesh2D, k0::Float64, chi::AbstractVector{Float64},
        E_inc::AbstractVector{ComplexF64};
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    length(E_inc) == mesh.ncells ||
        throw(DimensionMismatch(
            "E_inc length $(length(E_inc)) must match $(mesh.ncells) mesh cells."))
    all(isfinite, E_inc) ||
        throw(ArgumentError("E_inc must contain only finite values."))

    Z, D = assemble_vie_2d(
        mesh, k0, chi; max_output_bytes=max_output_bytes)
    Z_lu = _factor_vie_system_2d(Z, "solve_vie_2d")
    issuccess(Z_lu) ||
        error("solve_vie_2d system matrix factorization failed.")
    E_total = _solve_factored_linear_system(
        Z_lu, Z, E_inc, "solve_vie_2d")
    all(isfinite, E_total) ||
        error("solve_vie_2d produced non-finite total-field values.")

    return VIEResult2D(E_total, Vector(E_inc), Vector(chi), D, Z, Z_lu, mesh, k0)
end
