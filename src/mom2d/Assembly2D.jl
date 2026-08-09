# Assembly2D.jl — 2D TM Volume Integral Equation assembly and solve
#
# VIE for TM scattering from inhomogeneous dielectric:
#   E_z(r) = E_z^inc(r) + k₀² ∫_D χ(r') G₂D(r,r') E_z(r') dA'
#
# MoM discretization (pulse basis, point matching):
#   Z E = E^inc,  where Z = I - k₀² D diag(χ)   (χ_n scales column n: Z[m,n] = δ - k₀² χ_n D[m,n])

export assemble_vie_2d, solve_vie_2d

const _VIE_PRODUCT_FALLBACK_PRECISION_2D = 256

@inline _complex_component_scale_2d(value::ComplexF64) =
    max(abs(real(value)), abs(imag(value)))

@inline function _usable_nonzero_product_2d(value::ComplexF64)
    isfinite(value) || return false
    scale = _complex_component_scale_2d(value)
    return scale >= floatmin(Float64)
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
    k0sq = k0^2
    isfinite(k0sq) ||
        throw(ArgumentError(
            "assemble_vie_2d squared wavenumber must be finite, got $k0sq."))

    D = _assemble_D_matrix_unchecked(mesh, k0)
    N = mesh.ncells

    Z = Matrix{ComplexF64}(undef, N, N)

    @inbounds for n in 1:N
        for m in 1:N
            Z[m, n] = -_range_safe_product_2d(
                k0sq, chi[n], D[m, n],
                "assemble_vie_2d interaction coefficient")
        end
        Z[n, n] += 1.0  # add identity
    end

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
    E_total = Z_lu \ E_inc
    all(isfinite, E_total) ||
        error("solve_vie_2d produced non-finite total-field values.")

    return VIEResult2D(E_total, Vector(E_inc), Vector(chi), D, Z, Z_lu, mesh, k0)
end
