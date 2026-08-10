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

function _solve_vie_factored_2d(
        factorization,
        E_inc::AbstractVector{ComplexF64})
    rhs_scale = 0.0
    @inbounds for value in E_inc
        rhs_scale = max(rhs_scale, _complex_component_scale_2d(value))
    end

    if !iszero(rhs_scale) &&
       (rhs_scale < _VIE_RHS_SCALE_LOWER_2D ||
        rhs_scale > _VIE_RHS_SCALE_UPPER_2D)
        E_total = Vector(E_inc)
        E_total ./= rhs_scale
        ldiv!(factorization, E_total)
        @inbounds for index in eachindex(E_total)
            E_total[index] *= rhs_scale
        end
        return E_total
    end
    return factorization \ E_inc
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
    assemble_vie_2d(mesh, k0, chi)

Assemble VIE system matrix: Z[m,n] = δ[m,n] - k₀² χ[n] D[m,n]

Returns (Z, D) where D is the Green's function integral matrix.
"""
function assemble_vie_2d(mesh::Mesh2D, k0::Float64, chi::AbstractVector{Float64})
    _validate_mesh_2d(mesh)
    _validate_positive_finite_2d(k0, "assemble_vie_2d wavenumber")
    length(chi) == mesh.ncells ||
        throw(DimensionMismatch(
            "chi length $(length(chi)) must match $(mesh.ncells) mesh cells."))
    all(isfinite, chi) ||
        throw(ArgumentError("chi must contain only finite values."))
    N = mesh.ncells
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
    solve_vie_2d(mesh, k0, chi, E_inc)

Solve the 2D VIE for internal total fields.
Returns `VIEResult2D` with all computed quantities for downstream use.
"""
function solve_vie_2d(mesh::Mesh2D, k0::Float64, chi::AbstractVector{Float64},
                      E_inc::AbstractVector{ComplexF64})
    length(E_inc) == mesh.ncells ||
        throw(DimensionMismatch(
            "E_inc length $(length(E_inc)) must match $(mesh.ncells) mesh cells."))
    all(isfinite, E_inc) ||
        throw(ArgumentError("E_inc must contain only finite values."))

    Z, D = assemble_vie_2d(mesh, k0, chi)
    Z_lu = lu(Z)
    issuccess(Z_lu) ||
        error("solve_vie_2d system matrix factorization failed.")
    E_total = _solve_vie_factored_2d(Z_lu, E_inc)
    all(isfinite, E_total) ||
        error("solve_vie_2d produced non-finite total-field values.")

    return VIEResult2D(E_total, Vector(E_inc), Vector(chi), D, Z, Z_lu, mesh, k0)
end
