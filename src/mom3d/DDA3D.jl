# DDA3D.jl -- 3D vector material scattering by discrete dipoles
#
# Convention: exp(+i omega t), matching the rest of the package.
# The scalar Green phase is exp(-i k R). The electric field from a normalized
# electric dipole q = p / eps0 is
#
#   E(r) = G_EE(r, r') q
#
# where G_EE is the free-space electric dipole dyadic without the 1/eps0 factor.

export clausius_mossotti_polarizability, dda_polarizabilities
export dda_operator_3d
export electric_dipole_dyadic_3d, assemble_dda_3d, solve_dda_3d
export planewave_dda_3d, induced_dipoles_dda_3d
export scattered_field_dda_3d, farfield_dda_3d

const _I3_DDA = @SMatrix [1.0 0.0 0.0;
                          0.0 1.0 0.0;
                          0.0 0.0 1.0]

const _CI3_DDA = @SMatrix [1.0 + 0im 0.0 + 0im 0.0 + 0im;
                           0.0 + 0im 1.0 + 0im 0.0 + 0im;
                           0.0 + 0im 0.0 + 0im 1.0 + 0im]
const _CMat3DDA = SMatrix{3,3,ComplexF64,9}
# A floatmax-scale sum can cancel to the smallest Float64 subnormal, spanning
# 2099 binary orders of magnitude. Keep a guard margin for products and sums.
const _DDA_ACCUMULATION_FALLBACK_PRECISION = 2304
# A component of αv + βy contains four products of Float64 components. Their
# exact binary terms can span bit positions -2148 through 2049, requiring at
# most 4198 significant bits after cancellation. Keep a guard margin while
# confining the allocation-heavy path to exceptional exponent ranges.
const _DDA_SCALED_OUTPUT_FALLBACK_PRECISION = 4352

@inline _dda_index(voxel::Int, comp::Int) = 3 * (voxel - 1) + comp

@inline _dda_voxel(linear_index::Int) = div(linear_index - 1, 3) + 1
@inline _dda_component(linear_index::Int) = mod(linear_index - 1, 3) + 1

@noinline function _dda_scaled_output_bigfloat_3d(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    return setprecision(
            BigFloat, _DDA_SCALED_OUTPUT_FALLBACK_PRECISION) do
        total = Complex{BigFloat}(alpha_scale) *
                Complex{BigFloat}(value)
        if !overwrite
            total += Complex{BigFloat}(beta_scale) *
                     Complex{BigFloat}(previous)
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "DDA-family scaled output is outside the representable " *
                "ComplexF64 range at row $row."))
        return converted
    end
end

@inline function _dda_scaled_output_3d(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    alpha_term = alpha_scale * value
    if overwrite
        converted = ComplexF64(alpha_term)
        return isfinite(converted) ? converted :
               _dda_scaled_output_bigfloat_3d(
                   value, previous, alpha_scale, beta_scale, true, row)
    end

    beta_term = beta_scale * previous
    combined = alpha_term + beta_term
    magnitude_sum =
        max(abs(real(alpha_term)), abs(imag(alpha_term))) +
        max(abs(real(beta_term)), abs(imag(beta_term)))
    converted = ComplexF64(combined)
    if isfinite(converted) && isfinite(magnitude_sum)
        return converted
    end
    return _dda_scaled_output_bigfloat_3d(
        value, previous, alpha_scale, beta_scale, false, row)
end

@inline function _finite_nonnegative_k0_3d(k0::Real)
    k = Float64(k0)
    isfinite(k) && k >= 0 ||
        throw(ArgumentError("k0 must be finite and nonnegative, got $k0."))
    return k
end

@inline function _finite_positive_k0_3d(k0::Real)
    k = Float64(k0)
    isfinite(k) && k > 0 ||
        throw(ArgumentError("k0 must be finite and positive, got $k0."))
    return k
end

@inline function _finite_positive_real_3d(value::Real, label::AbstractString)
    converted = Float64(value)
    isfinite(converted) && converted > 0 ||
        throw(ArgumentError(
            "$label must be finite and positive, got $value."))
    return converted
end

@inline function _read_field_component(x::AbstractVector, j::Int)
    return CVec3(x[_dda_index(j, 1)], x[_dda_index(j, 2)], x[_dda_index(j, 3)])
end

function _as_cvec3(v, label::AbstractString)
    length(v) == 3 || error("$label must have exactly three components.")
    out = CVec3(ComplexF64(v[1]), ComplexF64(v[2]), ComplexF64(v[3]))
    for c in out
        isfinite(real(c)) && isfinite(imag(c)) ||
            error("$label contains a non-finite component: $c.")
    end
    return out
end

@inline function _normalized_real_direction_dda_3d(
    value::Vec3,
    label::AbstractString,
)
    all(isfinite, value) ||
        throw(ArgumentError("$label components must be finite."))
    scale = max(abs(value[1]), abs(value[2]), abs(value[3]))
    scale > 0.0 ||
        throw(ArgumentError("$label must be nonzero."))
    scaled = value / scale
    scaled_norm = norm(scaled)
    isfinite(scaled_norm) && scaled_norm > 0.0 ||
        throw(ArgumentError("$label must have a finite, nonzero norm."))
    return scaled / scaled_norm
end

@inline function _normalized_complex_direction_dda_3d(
    value::CVec3,
    label::AbstractString,
)
    scale = 0.0
    @inbounds for component in value
        scale = max(scale, abs(real(component)), abs(imag(component)))
    end
    scale > 0.0 ||
        throw(ArgumentError("$label must be nonzero."))
    scaled = value / scale
    scaled_norm = norm(scaled)
    isfinite(scaled_norm) && scaled_norm > 0.0 ||
        throw(ArgumentError("$label must have a finite, nonzero norm."))
    return scaled / scaled_norm
end

@inline function _farfield_prefactor_dda_3d(k::Float64)
    prefactor = k * k / (4π)
    if isfinite(prefactor) && prefactor > 0.0
        return prefactor, 0.0, 0
    end

    k_fraction, k_exponent = frexp(k)
    fraction, exponent = frexp(k_fraction * k_fraction / (4π))
    return prefactor, fraction, 2k_exponent + exponent
end

@inline function _scale_farfield_component_dda_3d(
    value::ComplexF64,
    fraction::Float64,
    exponent::Int,
)
    real_fraction, real_exponent = frexp(real(value))
    imag_fraction, imag_exponent = frexp(imag(value))
    return ComplexF64(
        ldexp(real_fraction * fraction, real_exponent + exponent),
        ldexp(imag_fraction * fraction, imag_exponent + exponent),
    )
end

@inline function _scale_farfield_vector_dda_3d(
    value::CVec3,
    fraction::Float64,
    exponent::Int,
)
    return CVec3(
        _scale_farfield_component_dda_3d(value[1], fraction, exponent),
        _scale_farfield_component_dda_3d(value[2], fraction, exponent),
        _scale_farfield_component_dda_3d(value[3], fraction, exponent),
    )
end

function _copy_finite_complex_vector_3d(
    values,
    expected_length::Int,
    label::AbstractString,
)
    length(values) == expected_length ||
        throw(DimensionMismatch(
            "$label length ($(length(values))) must be $expected_length."))
    out = Vector{ComplexF64}(undef, expected_length)
    @inbounds for (j, value) in enumerate(values)
        converted = try
            ComplexF64(value)
        catch err
            err isa Union{InexactError,MethodError,OverflowError} || rethrow()
            throw(ArgumentError(
                "$label[$j] cannot be converted to ComplexF64."))
        end
        isfinite(converted) ||
            throw(ArgumentError(
                "$label[$j] must be finite, got $converted."))
        out[j] = converted
    end
    return out
end

function _flatten_fields_3d(fields::AbstractVector, n::Int, label::AbstractString)
    length(fields) == n || error("$label length ($(length(fields))) must match nvoxels ($n).")
    out = Vector{ComplexF64}(undef, 3n)
    for j in 1:n
        v = _as_cvec3(fields[j], "$label[$j]")
        out[_dda_index(j, 1)] = v[1]
        out[_dda_index(j, 2)] = v[2]
        out[_dda_index(j, 3)] = v[3]
    end
    return out
end

function _unflatten_fields_3d(x::AbstractVector{ComplexF64}, n::Int)
    length(x) == 3n || error("field vector length ($(length(x))) must be 3*nvoxels ($(3n)).")
    out = Vector{CVec3}(undef, n)
    for j in 1:n
        out[j] = CVec3(x[_dda_index(j, 1)], x[_dda_index(j, 2)], x[_dda_index(j, 3)])
    end
    return out
end

function _coerce_epsr_3d(eps_r, n::Int)
    if eps_r isa Number
        epsc = ComplexF64(eps_r)
        isfinite(epsc) ||
            throw(ArgumentError("eps_r must be finite, got $eps_r."))
        return fill(epsc, n)
    else
        return _copy_finite_complex_vector_3d(eps_r, n, "eps_r")
    end
end

function _as_cmat3(m, label::AbstractString)
    size(m) == (3, 3) || error("$label must be a 3x3 tensor.")
    vals = ntuple(i -> ComplexF64(m[i]), 9)
    M = _CMat3DDA(vals)
    for c in M
        isfinite(real(c)) && isfinite(imag(c)) ||
            error("$label contains a non-finite component: $c.")
    end
    return M
end

function _diag_tensor_from_tuple(t, label::AbstractString)
    length(t) == 3 || error("$label diagonal tensor tuple must have three entries.")
    vals = ComplexF64.(t)
    all(isfinite, vals) ||
        throw(ArgumentError("$label contains a non-finite component."))
    return _CMat3DDA((vals[1], 0, 0,
                      0, vals[2], 0,
                      0, 0, vals[3]))
end

_is_diag_tensor_tuple(x) = x isa Tuple && length(x) == 3 && all(v -> v isa Number, x)

function _coerce_epsr_material_3d(eps_r, n::Int)
    if eps_r isa Number
        return _coerce_epsr_3d(eps_r, n)
    elseif eps_r isa AbstractMatrix
        return fill(_as_cmat3(eps_r, "eps_r"), n)
    elseif _is_diag_tensor_tuple(eps_r)
        return fill(_diag_tensor_from_tuple(eps_r, "eps_r"), n)
    else
        length(eps_r) == n ||
            error("eps_r length ($(length(eps_r))) must match nvoxels ($n).")
        first_eps = eps_r[1]
        if first_eps isa Number
            return _coerce_epsr_3d(eps_r, n)
        elseif first_eps isa AbstractMatrix
            return [_as_cmat3(eps_r[j], "eps_r[$j]") for j in 1:n]
        elseif _is_diag_tensor_tuple(first_eps)
            return [_diag_tensor_from_tuple(eps_r[j], "eps_r[$j]") for j in 1:n]
        else
            error("Unsupported eps_r material specification: $(typeof(first_eps)).")
        end
    end
end

@inline _alpha_apply(alpha::Number, E::CVec3) = alpha * E
@inline _alpha_apply(alpha::_CMat3DDA, E::CVec3) = alpha * E
@inline _alpha_adjoint_apply(alpha::Number, E::CVec3) = conj(alpha) * E
@inline _alpha_adjoint_apply(alpha::_CMat3DDA, E::CVec3) = adjoint(alpha) * E
@inline _alpha_block(G, alpha::Number) = G * alpha
@inline _alpha_block(G, alpha::_CMat3DDA) = G * alpha

@inline function _alpha_basis_vector_3d(alpha::Number, column::Int)
    z = zero(ComplexF64)
    value = ComplexF64(alpha)
    return column == 1 ? CVec3(value, z, z) :
           column == 2 ? CVec3(z, value, z) : CVec3(z, z, value)
end

@inline function _alpha_basis_vector_3d(alpha::_CMat3DDA, column::Int)
    return CVec3(alpha[1, column], alpha[2, column], alpha[3, column])
end

# The fallback only receives binary64 inputs. Use more than four input
# significands for the small (at most 6x6) solves, while keeping this rare
# exponent-range recovery path bounded in time and memory.
const _POLARIZABILITY_FALLBACK_PRECISION = 256
const _POLARIZABILITY_SAFE_FACTOR_EXPONENT = 128

@inline function _polarizability_extreme_component_3d(value::Real)
    magnitude = abs(Float64(value))
    iszero(magnitude) && return false
    component_exponent = exponent(magnitude)
    return component_exponent < -_POLARIZABILITY_SAFE_FACTOR_EXPONENT ||
           component_exponent > _POLARIZABILITY_SAFE_FACTOR_EXPONENT
end

@inline function _radiative_correction_requires_bigfloat_3d(
        k::Float64, alpha::Number)
    iszero(k) && return false
    return _polarizability_extreme_component_3d(k) ||
           _polarizability_extreme_component_3d(real(alpha)) ||
           _polarizability_extreme_component_3d(imag(alpha))
end

@inline function _radiative_correction_requires_bigfloat_3d(
        k::Float64, alpha::AbstractArray)
    iszero(k) && return false
    _polarizability_extreme_component_3d(k) && return true
    @inbounds for value in alpha
        if _polarizability_extreme_component_3d(real(value)) ||
           _polarizability_extreme_component_3d(imag(value))
            return true
        end
    end
    return false
end

@noinline function _scaled_alpha_apply_bigfloat_3d(
        alpha::Number,
        field::SVector{N,ComplexF64},
        divisor::Float64,
        label::AbstractString,
        voxel::Int) where {N}
    value = setprecision(BigFloat, _POLARIZABILITY_FALLBACK_PRECISION) do
        alpha_big = Complex{BigFloat}(alpha)
        divisor_big = BigFloat(divisor)
        SVector{N,ComplexF64}(ntuple(index -> ComplexF64(
            alpha_big * Complex{BigFloat}(field[index]) / divisor_big), N))
    end
    all(isfinite, value) ||
        throw(OverflowError(
            "$label at voxel $voxel is outside the representable ComplexF64 range."))
    return value
end

@noinline function _scaled_alpha_apply_bigfloat_3d(
        alpha::SMatrix{N,N,ComplexF64,L},
        field::SVector{N,ComplexF64},
        divisor::Float64,
        label::AbstractString,
        voxel::Int) where {N,L}
    value = setprecision(BigFloat, _POLARIZABILITY_FALLBACK_PRECISION) do
        divisor_big = BigFloat(divisor)
        entries = ntuple(row -> begin
            accumulator = zero(Complex{BigFloat})
            @inbounds for column in 1:N
                accumulator += Complex{BigFloat}(alpha[row, column]) *
                               Complex{BigFloat}(field[column])
            end
            ComplexF64(accumulator / divisor_big)
        end, N)
        SVector{N,ComplexF64}(entries)
    end
    all(isfinite, value) ||
        throw(OverflowError(
            "$label at voxel $voxel is outside the representable ComplexF64 range."))
    return value
end

@inline function _scaled_alpha_apply_3d(
        alpha::Number,
        field::SVector{N,ComplexF64},
        divisor::Float64,
        label::AbstractString,
        voxel::Int) where {N}
    alpha_value = ComplexF64(alpha)
    scaled_alpha = alpha_value / divisor
    scale_lost = !iszero(alpha_value) && iszero(scaled_alpha)
    value = scaled_alpha * field
    !scale_lost && all(isfinite, value) && return value
    return _scaled_alpha_apply_bigfloat_3d(
        alpha_value, field, divisor, label, voxel)
end

@inline function _scaled_alpha_apply_3d(
        alpha::SMatrix{N,N,ComplexF64,L},
        field::SVector{N,ComplexF64},
        divisor::Float64,
        label::AbstractString,
        voxel::Int) where {N,L}
    scaled_alpha = alpha / divisor
    scale_lost = false
    @inbounds for index in eachindex(alpha)
        if !iszero(alpha[index]) && iszero(scaled_alpha[index])
            scale_lost = true
            break
        end
    end
    value = scaled_alpha * field
    !scale_lost && all(isfinite, value) && return value
    return _scaled_alpha_apply_bigfloat_3d(
        alpha, field, divisor, label, voxel)
end

function _clausius_mossotti_bigfloat(
    epsc::ComplexF64,
    V::Float64,
    k::Float64,
    radiative_correction::Bool,
)
    return setprecision(BigFloat, _POLARIZABILITY_FALLBACK_PRECISION) do
        epsb = Complex{BigFloat}(BigFloat(real(epsc)), BigFloat(imag(epsc)))
        alpha = 3 * BigFloat(V) * (epsb - 1) / (epsb + 2)
        if radiative_correction
            denominator = 1 + Complex{BigFloat}(0, 1) * BigFloat(k)^3 * alpha /
                              (6 * BigFloat(pi))
            iszero(denominator) &&
                error("Radiatively corrected Clausius-Mossotti polarizability is singular.")
            alpha /= denominator
        end
        converted = ComplexF64(alpha)
        isfinite(converted) ||
            throw(OverflowError(
                "Clausius-Mossotti polarizability is outside the representable ComplexF64 range."))
        return converted
    end
end

function _validated_clausius_mossotti_inverse_bigfloat_3d(
        denominator::AbstractMatrix{Complex{BigFloat}},
        label::AbstractString)
    n = size(denominator, 1)
    scale = maximum(component -> max(
        abs(real(component)), abs(imag(component))), denominator)
    tolerance = BigFloat(100 * eps(Float64))
    scale > tolerance || error("$label is singular or too close to singular.")
    normalized = denominator / scale
    factorization = lu(normalized; check=false)
    issuccess(factorization) ||
        error("$label is singular or too close to singular.")
    inverse_normalized = inv(factorization)
    inverse_norm = maximum(row ->
        sum(column -> abs(inverse_normalized[row, column]), 1:n), 1:n)
    scale / inverse_norm > tolerance ||
        error("$label is singular or too close to singular.")
    return inverse_normalized / scale
end

@inline function _validated_clausius_mossotti_inverse_3d(
    denominator::SMatrix{N,N,ComplexF64,L},
    label::AbstractString,
) where {N,L}
    scale = 0.0
    @inbounds for component in denominator
        scale = max(scale, abs(real(component)), abs(imag(component)))
    end
    tolerance = 100 * eps(Float64)
    scale > tolerance || error("$label is singular or too close to singular.")

    normalized = denominator / scale
    factorization = lu(normalized; check=false)
    issuccess(factorization) || return nothing
    inverse_normalized = inv(factorization)
    all(isfinite, inverse_normalized) || return nothing

    inverse_norm = 0.0
    @inbounds for row in 1:N
        row_sum = 0.0
        for col in 1:N
            row_sum += abs(inverse_normalized[row, col])
        end
        inverse_norm = max(inverse_norm, row_sum)
    end
    scale / inverse_norm > tolerance ||
        error("$label is singular or too close to singular.")
    inverse_denominator = inverse_normalized / scale
    all(isfinite, inverse_denominator) || return nothing
    return inverse_denominator
end

function _clausius_mossotti_bigfloat(
    epsm::AbstractMatrix{ComplexF64},
    V::Float64,
    k::Float64,
    radiative_correction::Bool,
    label::AbstractString,
)
    return setprecision(BigFloat, _POLARIZABILITY_FALLBACK_PRECISION) do
        n = size(epsm, 1)
        epsb = Complex{BigFloat}.(epsm)
        identity_b = Matrix{Complex{BigFloat}}(I, n, n)
        denominator = epsb + 2 * identity_b
        inverse_denominator =
            _validated_clausius_mossotti_inverse_bigfloat_3d(
                denominator, "$label denominator")
        alpha = 3 * BigFloat(V) *
                ((epsb - identity_b) * inverse_denominator)
        if radiative_correction
            denominator = identity_b +
                          Complex{BigFloat}(0, 1) * BigFloat(k)^3 * alpha /
                          (6 * BigFloat(pi))
            alpha /= denominator
        end
        converted = ComplexF64.(alpha)
        all(isfinite, converted) ||
            throw(OverflowError(
                "$label is outside the representable ComplexF64 range."))
        return converted
    end
end

"""
    clausius_mossotti_polarizability(eps_r, volume; k0=0, radiative_correction=false)

Return normalized electric polarizability `alpha = p / (eps0 * E)` for an
isotropic voxel of relative permittivity `eps_r` and volume `volume`.

The default is the Clausius-Mossotti polarizability

    alpha0 = 3V * (eps_r - 1) / (eps_r + 2)

which gives the exact electrostatic polarizability of a sphere with the same
volume. If `radiative_correction=true`, a leading-order radiation-reaction
correction consistent with the package's `exp(+i omega t)` convention is used.
"""
function clausius_mossotti_polarizability(eps_r::Number, volume::Real;
                                          k0::Real=0.0,
                                          radiative_correction::Bool=false)
    V = _finite_positive_real_3d(volume, "volume")
    k = _finite_nonnegative_k0_3d(k0)
    epsc = ComplexF64(eps_r)
    isfinite(epsc) ||
        throw(ArgumentError("eps_r must be finite, got $eps_r."))
    abs(epsc + 2) > 100 * eps(Float64) ||
        error("Clausius-Mossotti polarizability is singular for eps_r near -2.")
    # Apply the material ratio before the possibly large volume. This avoids
    # overflowing `3V` when the final polarizability is representable.
    alpha0 = V * (3 * ((epsc - 1) / (epsc + 2)))
    if isfinite(alpha0)
        if !radiative_correction
            return alpha0
        end
        _radiative_correction_requires_bigfloat_3d(k, alpha0) &&
            return _clausius_mossotti_bigfloat(
                epsc, V, k, radiative_correction)
        corrected = alpha0 / (1 + 1im * k^3 * alpha0 / (6π))
        isfinite(corrected) && return corrected
    end
    # This path is rare and retains representable results when exponent range,
    # rather than mathematical singularity, defeated binary64 intermediates.
    return _clausius_mossotti_bigfloat(epsc, V, k, radiative_correction)
end

function clausius_mossotti_polarizability(eps_r::AbstractMatrix, volume::Real;
                                          k0::Real=0.0,
                                          radiative_correction::Bool=false)
    V = _finite_positive_real_3d(volume, "volume")
    k = _finite_nonnegative_k0_3d(k0)
    epsm = _as_cmat3(eps_r, "eps_r")
    denom = epsm + 2 * _CI3_DDA
    inverse_denom = _validated_clausius_mossotti_inverse_3d(
        denom, "Tensor Clausius-Mossotti denominator eps_r + 2I")
    inverse_denom === nothing &&
        return _CMat3DDA(Tuple(_clausius_mossotti_bigfloat(
            epsm, V, k, radiative_correction,
            "Tensor Clausius-Mossotti polarizability")))
    alpha0 = V * (3 * ((epsm - _CI3_DDA) * inverse_denom))
    if all(isfinite, alpha0)
        if !radiative_correction
            return alpha0
        end
        _radiative_correction_requires_bigfloat_3d(k, alpha0) &&
            return _CMat3DDA(Tuple(_clausius_mossotti_bigfloat(
                epsm, V, k, radiative_correction,
                "Tensor Clausius-Mossotti polarizability")))
        corrected = alpha0 / (_CI3_DDA + 1im * k^3 * alpha0 / (6π))
        all(isfinite, corrected) && return corrected
    end
    converted = _clausius_mossotti_bigfloat(
        epsm, V, k, radiative_correction,
        "Tensor Clausius-Mossotti polarizability")
    return _CMat3DDA(Tuple(converted))
end

"""
    dda_polarizabilities(grid, k0, eps_r; radiative_correction=false)

Compute normalized electric polarizability for every voxel in `grid`.
"""
function dda_polarizabilities(grid::VoxelGrid3D, k0::Real, eps_r;
                              radiative_correction::Bool=false)
    k = _finite_nonnegative_k0_3d(k0)
    epsv = _coerce_epsr_material_3d(eps_r, grid.nvoxels)
    return _dda_polarizabilities_from_coerced(
        grid, k, epsv, radiative_correction)
end

function _dda_polarizabilities_from_coerced(
    grid::VoxelGrid3D,
    k::Float64,
    epsv::AbstractVector,
    radiative_correction::Bool,
)
    length(epsv) == grid.nvoxels ||
        throw(DimensionMismatch(
            "eps_r length ($(length(epsv))) must match nvoxels ($(grid.nvoxels))."))
    first_alpha = clausius_mossotti_polarizability(
        epsv[1], grid.volumes[1];
        k0=k,
        radiative_correction=radiative_correction,
    )
    alpha = Vector{typeof(first_alpha)}(undef, grid.nvoxels)
    alpha[1] = first_alpha
    @inbounds for j in 2:grid.nvoxels
        alpha[j] = clausius_mossotti_polarizability(
            epsv[j], grid.volumes[j];
            k0=k,
            radiative_correction=radiative_correction,
        )
    end
    return alpha
end

"""
    dda_operator_3d(grid, k0, eps_r; radiative_correction=false)

Construct the matrix-free DDA material operator. This is the memory-efficient
counterpart to `assemble_dda_3d`.
"""
function dda_operator_3d(grid::VoxelGrid3D, k0::Real, eps_r;
                         radiative_correction::Bool=false)
    k = _finite_positive_k0_3d(k0)
    epsv = _coerce_epsr_material_3d(eps_r, grid.nvoxels)
    alpha = _dda_polarizabilities_from_coerced(
        grid, k, epsv, radiative_correction)
    return DDAOperator3D(grid, k, epsv, alpha, radiative_correction)
end

"""
    electric_dipole_dyadic_3d(r, rp, k0)

Free-space electric dipole dyadic for the package convention `exp(+i omega t)`.
It maps normalized dipole moment `q = p / eps0` at `rp` to electric field at
`r`. The singular self term is not defined and must be handled by the chosen
polarizability model.
"""
@inline function _electric_dipole_geometry_3d(r::Vec3, rp::Vec3, k::Float64)
    all(isfinite, r) && all(isfinite, rp) ||
        throw(ArgumentError(
            "electric dipole observation and source points must be finite."))
    R_vec = r - rp
    R = hypot(R_vec[1], R_vec[2], R_vec[3])
    isfinite(R) ||
        throw(OverflowError(
            "electric dipole point separation is outside the representable Float64 range."))
    R > 0 ||
        error("electric_dipole_dyadic_3d requires distinct finite points.")
    Rhat = R_vec / R
    phase = k * R
    isfinite(phase) ||
        throw(OverflowError(
            "electric dipole phase k0*R is outside the representable Float64 range."))
    return R, Rhat, phase, cis(-phase) / (4π)
end

@inline function _electric_dipole_geometry_requires_exact_3d(
    r::Vec3,
    rp::Vec3,
    k::Float64,
)
    all(isfinite, r) && all(isfinite, rp) ||
        throw(ArgumentError(
            "electric dipole observation and source points must be finite."))
    displacement = r - rp
    distance = hypot(displacement[1], displacement[2], displacement[3])
    iszero(distance) &&
        error("electric_dipole_dyadic_3d requires distinct finite points.")
    !isfinite(distance) && return true

    # A finite Float64 phase can still be unrelated to the phase of the exact
    # supplied coordinates once its ulp is appreciable.  Extreme coordinates
    # also require exact subtraction before the distance is formed.
    @inbounds for component in 1:3
        (_source_scaling_extreme_value(r[component]) ||
         _source_scaling_extreme_value(rp[component])) && return true
    end
    _source_scaling_extreme_value(k) && return true
    phase = k * distance
    return !isfinite(phase) ||
           (!iszero(phase) && exponent(abs(phase)) > 32)
end

@inline _electric_dipole_exact_precision_3d(
    r::Vec3,
    rp::Vec3,
    k::Float64,
) = _source_radial_phase_precision(k, r, rp)

@inline function _scale_dipole_component_3d(value::Float64,
                                             amplitude::Float64,
                                             k::Float64, k_power::Int,
                                             R::Float64, R_power::Int)
    (iszero(value) || iszero(amplitude) || (k_power > 0 && iszero(k))) &&
        return copysign(0.0, value)

    value_fraction, value_exponent = frexp(value)
    amplitude_fraction, amplitude_exponent = frexp(amplitude)
    R_fraction, R_exponent = frexp(R)
    fraction = value_fraction * amplitude_fraction
    exponent = value_exponent + amplitude_exponent - R_power * R_exponent

    if k_power > 0
        k_fraction, k_exponent = frexp(k)
        @inbounds for _ in 1:k_power
            fraction *= k_fraction
        end
        exponent += k_power * k_exponent
    end
    @inbounds for _ in 1:R_power
        fraction /= R_fraction
    end
    return ldexp(fraction, exponent)
end

@inline function _scale_dipole_value_3d(value::ComplexF64,
                                         amplitude::Float64,
                                         k::Float64, k_power::Int,
                                         R::Float64, R_power::Int)
    return ComplexF64(
        _scale_dipole_component_3d(
            real(value), amplitude, k, k_power, R, R_power),
        _scale_dipole_component_3d(
            imag(value), amplitude, k, k_power, R, R_power),
    )
end

@inline function _scale_dipole_vector_3d(value::CVec3,
                                          amplitude::Float64,
                                          k::Float64, k_power::Int,
                                          R::Float64, R_power::Int)
    return CVec3(
        _scale_dipole_value_3d(value[1], amplitude, k, k_power, R, R_power),
        _scale_dipole_value_3d(value[2], amplitude, k, k_power, R, R_power),
        _scale_dipole_value_3d(value[3], amplitude, k, k_power, R, R_power),
    )
end

@inline _complex_component_scale_3d(value::ComplexF64) =
    max(abs(real(value)), abs(imag(value)))

@inline function _floating_product_loses_range_3d(
        left::ComplexF64,
        right::ComplexF64)
    (iszero(left) || iszero(right)) && return false
    # Inspect primitive components before forming the complex product.  A
    # normal real component can otherwise hide an imaginary product that has
    # rounded to zero; a later Green-function or far-field scale can make that
    # lost component representable again.
    (_ieee_dense_extreme_factor(left, Float64) ||
     _ieee_dense_extreme_factor(right, Float64)) && return true
    product = left * right
    isfinite(product) || return true
    scale = _complex_component_scale_3d(product)
    return iszero(scale) || scale < floatmin(Float64)
end

@inline function _alpha_field_product_loses_range_3d(
        alpha::Number,
        field::SVector{N,ComplexF64}) where {N}
    coefficient = ComplexF64(alpha)
    @inbounds for component in field
        _floating_product_loses_range_3d(coefficient, component) && return true
    end
    return false
end

@inline function _alpha_field_product_loses_range_3d(
        alpha::SMatrix{N,N,ComplexF64,L},
        field::SVector{N,ComplexF64}) where {N,L}
    @inbounds for column in 1:N, row in 1:N
        _floating_product_loses_range_3d(
            alpha[row, column], field[column]) && return true
    end
    return false
end

@inline function _alpha_apply_bigfloat_vector_3d(
        alpha::Number,
        field::SVector{N,ComplexF64}) where {N}
    coefficient = Complex{BigFloat}(alpha)
    return SVector{N,Complex{BigFloat}}(ntuple(
        index -> coefficient * Complex{BigFloat}(field[index]), N))
end

@inline function _alpha_apply_bigfloat_vector_3d(
        alpha::SMatrix{N,N,ComplexF64,L},
        field::SVector{N,ComplexF64}) where {N,L}
    return SVector{N,Complex{BigFloat}}(ntuple(row -> begin
        accumulator = zero(Complex{BigFloat})
        @inbounds for column in 1:N
            accumulator += Complex{BigFloat}(alpha[row, column]) *
                           Complex{BigFloat}(field[column])
        end
        accumulator
    end, N))
end

@inline function _alpha_adjoint_apply_bigfloat_vector_3d(
        alpha::Number,
        field::SVector{N,Complex{BigFloat}}) where {N}
    coefficient = conj(Complex{BigFloat}(alpha))
    return SVector{N,Complex{BigFloat}}(ntuple(
        index -> coefficient * field[index], N))
end

@inline function _alpha_adjoint_apply_bigfloat_vector_3d(
        alpha::SMatrix{N,N,ComplexF64,L},
        field::SVector{N,Complex{BigFloat}}) where {N,L}
    return SVector{N,Complex{BigFloat}}(ntuple(row -> begin
        accumulator = zero(Complex{BigFloat})
        @inbounds for column in 1:N
            accumulator += conj(Complex{BigFloat}(alpha[column, row])) *
                           field[column]
        end
        accumulator
    end, N))
end

function _electric_dipole_value_bigfloat_3d(
        r::Vec3,
        rp::Vec3,
        k::Float64,
        q::SVector{3,Complex{BigFloat}})
    R_vec = SVector{3,BigFloat}(
        BigFloat(r[1]) - BigFloat(rp[1]),
        BigFloat(r[2]) - BigFloat(rp[2]),
        BigFloat(r[3]) - BigFloat(rp[3]),
    )
    R = sqrt(sum(abs2, R_vec))
    R > 0 || error(
        "electric_dipole_dyadic_3d requires distinct finite points.")
    Rhat = R_vec / R
    rq = sum(Rhat[index] * q[index] for index in 1:3)
    transverse_vector = q - rq * Rhat
    near_vector = 3 * rq * Rhat - q
    kb = BigFloat(k)
    expfac = exp(Complex{BigFloat}(0, -kb * R)) /
             (4 * BigFloat(pi))
    return expfac * (
        (kb^2 / R) * transverse_vector +
        (inv(R^3) + Complex{BigFloat}(0, 1) * kb / R^2) * near_vector)
end

function _electric_dipole_apply_bigfloat_3d(r::Vec3, rp::Vec3, k::Float64,
                                             q::CVec3)
    precision = _electric_dipole_exact_precision_3d(r, rp, k)
    return setprecision(BigFloat, precision) do
        qb = SVector{3,Complex{BigFloat}}(
            Complex{BigFloat}(q[1]),
            Complex{BigFloat}(q[2]),
            Complex{BigFloat}(q[3]),
        )
        value = _electric_dipole_value_bigfloat_3d(r, rp, k, qb)
        converted = CVec3(
            ComplexF64(value[1]),
            ComplexF64(value[2]),
            ComplexF64(value[3]),
        )
        all(isfinite, converted) ||
            throw(OverflowError(
                "electric dipole field is outside the representable ComplexF64 range."))
        return converted
    end
end

function _electric_dipole_alpha_apply_bigfloat_3d(
        r::Vec3,
        rp::Vec3,
        k::Float64,
        alpha,
        field::CVec3)
    precision = _electric_dipole_exact_precision_3d(r, rp, k)
    return setprecision(BigFloat, precision) do
        q = _alpha_apply_bigfloat_vector_3d(alpha, field)
        value = _electric_dipole_value_bigfloat_3d(r, rp, k, q)
        converted = CVec3(
            ComplexF64(value[1]),
            ComplexF64(value[2]),
            ComplexF64(value[3]),
        )
        all(isfinite, converted) ||
            throw(OverflowError(
                "DDA interaction field is outside the representable ComplexF64 range."))
        return converted
    end
end

@inline function _electric_dipole_alpha_apply_3d(
        r::Vec3,
        rp::Vec3,
        k::Float64,
        alpha,
        field::CVec3)
    q = _alpha_apply(alpha, field)
    if all(isfinite, q) &&
       !_alpha_field_product_loses_range_3d(alpha, field)
        return _electric_dipole_apply_3d(r, rp, k, q)
    end
    return _electric_dipole_alpha_apply_bigfloat_3d(
        r, rp, k, alpha, field)
end

function _electric_dipole_alpha_adjoint_apply_bigfloat_3d(
        r::Vec3,
        rp::Vec3,
        k::Float64,
        alpha::_CMat3DDA,
        value::CVec3)
    precision = _electric_dipole_exact_precision_3d(r, rp, k)
    return setprecision(BigFloat, precision) do
        conjugated_value = SVector{3,Complex{BigFloat}}(ntuple(
            index -> conj(Complex{BigFloat}(value[index])), 3))
        green_conjugated = _electric_dipole_value_bigfloat_3d(
            r, rp, k, conjugated_value)
        green_adjoint = SVector{3,Complex{BigFloat}}(ntuple(
            index -> conj(green_conjugated[index]), 3))
        result = _alpha_adjoint_apply_bigfloat_vector_3d(
            alpha, green_adjoint)
        converted = CVec3(
            ComplexF64(result[1]),
            ComplexF64(result[2]),
            ComplexF64(result[3]),
        )
        all(isfinite, converted) ||
            throw(OverflowError(
                "adjoint DDA interaction is outside the representable ComplexF64 range."))
        return converted
    end
end

@inline function _electric_dipole_apply_with_geometry_3d(
    r::Vec3,
    rp::Vec3,
    k::Float64,
    q::CVec3,
    R::Float64,
    Rhat::Vec3,
    expfac::ComplexF64,
)
    q_scale = max(abs(q[1]), abs(q[2]), abs(q[3]))
    if !isfinite(q_scale)
        all(isfinite, q) ||
            throw(ArgumentError("electric dipole moment must be finite."))
        q_scale = max(
            _complex_component_scale_3d(q[1]),
            _complex_component_scale_3d(q[2]),
            _complex_component_scale_3d(q[3]),
        )
    end
    iszero(q_scale) && return CVec3(
        0.0 + 0im, 0.0 + 0im, 0.0 + 0im)

    rq_direct = dot(Rhat, q)
    ordinary_value = expfac * (
        (k^2 / R) * (q - rq_direct * Rhat) +
        (1 / R^3 + 1im * k / R^2) * (3 * rq_direct * Rhat - q))
    all(isfinite, ordinary_value) && return ordinary_value

    q_normalized = q / q_scale
    rq = dot(Rhat, q_normalized)
    transverse_vector = q_normalized - rq * Rhat
    near_vector = 3 * rq * Rhat - q_normalized
    transverse = _scale_dipole_vector_3d(
        transverse_vector, q_scale, k, 2, R, 1)
    near_real = _scale_dipole_vector_3d(
        near_vector, q_scale, k, 0, R, 3)
    near_imag = _scale_dipole_vector_3d(
        near_vector, q_scale, k, 1, R, 2)
    value = expfac * (transverse + near_real + 1im * near_imag)
    all(isfinite, value) && return value
    return _electric_dipole_apply_bigfloat_3d(r, rp, k, q)
end

@inline function _electric_dipole_alpha_block_3d(
    r::Vec3,
    rp::Vec3,
    k::Float64,
    alpha,
)
    if _electric_dipole_geometry_requires_exact_3d(r, rp, k)
        column1 = _electric_dipole_apply_bigfloat_3d(
            r, rp, k, _alpha_basis_vector_3d(alpha, 1))
        column2 = _electric_dipole_apply_bigfloat_3d(
            r, rp, k, _alpha_basis_vector_3d(alpha, 2))
        column3 = _electric_dipole_apply_bigfloat_3d(
            r, rp, k, _alpha_basis_vector_3d(alpha, 3))
        return _CMat3DDA((
            column1[1], column1[2], column1[3],
            column2[1], column2[2], column2[3],
            column3[1], column3[2], column3[3],
        ))
    end
    R, Rhat, _, expfac = _electric_dipole_geometry_3d(r, rp, k)
    column1 = _electric_dipole_apply_with_geometry_3d(
        r, rp, k, _alpha_basis_vector_3d(alpha, 1), R, Rhat, expfac)
    column2 = _electric_dipole_apply_with_geometry_3d(
        r, rp, k, _alpha_basis_vector_3d(alpha, 2), R, Rhat, expfac)
    column3 = _electric_dipole_apply_with_geometry_3d(
        r, rp, k, _alpha_basis_vector_3d(alpha, 3), R, Rhat, expfac)
    return _CMat3DDA((
        column1[1], column1[2], column1[3],
        column2[1], column2[2], column2[3],
        column3[1], column3[2], column3[3],
    ))
end

@inline function _electric_dipole_alpha_adjoint_apply_3d(
    r::Vec3,
    rp::Vec3,
    k::Float64,
    alpha::Number,
    value::CVec3,
)
    conjugated_value = CVec3(
        conj(value[1]), conj(value[2]), conj(value[3]))
    result = _electric_dipole_alpha_apply_3d(
        r, rp, k, alpha, conjugated_value)
    return CVec3(conj(result[1]), conj(result[2]), conj(result[3]))
end

@inline function _electric_dipole_alpha_adjoint_apply_3d(
    r::Vec3,
    rp::Vec3,
    k::Float64,
    alpha::_CMat3DDA,
    value::CVec3,
)
    _electric_dipole_geometry_requires_exact_3d(r, rp, k) &&
        return _electric_dipole_alpha_adjoint_apply_bigfloat_3d(
            r, rp, k, alpha, value)
    try
        R, Rhat, _, expfac = _electric_dipole_geometry_3d(r, rp, k)
        column1 = _electric_dipole_apply_with_geometry_3d(
            r, rp, k, _alpha_basis_vector_3d(alpha, 1), R, Rhat, expfac)
        column2 = _electric_dipole_apply_with_geometry_3d(
            r, rp, k, _alpha_basis_vector_3d(alpha, 2), R, Rhat, expfac)
        column3 = _electric_dipole_apply_with_geometry_3d(
            r, rp, k, _alpha_basis_vector_3d(alpha, 3), R, Rhat, expfac)
        result = CVec3(
            dot(column1, value),
            dot(column2, value),
            dot(column3, value),
        )
        all(isfinite, result) && return result
    catch err
        err isa OverflowError || rethrow()
    end
    return _electric_dipole_alpha_adjoint_apply_bigfloat_3d(
        r, rp, k, alpha, value)
end

function electric_dipole_dyadic_3d(r::Vec3, rp::Vec3, k0::Real)
    k = _finite_nonnegative_k0_3d(k0)
    return _electric_dipole_alpha_block_3d(r, rp, k, 1.0 + 0im)
end

@inline function _electric_dipole_apply_3d(r::Vec3, rp::Vec3, k::Float64, q::CVec3)
    _electric_dipole_geometry_requires_exact_3d(r, rp, k) &&
        return _electric_dipole_apply_bigfloat_3d(r, rp, k, q)
    R, Rhat, _, expfac = _electric_dipole_geometry_3d(r, rp, k)
    return _electric_dipole_apply_with_geometry_3d(
        r, rp, k, q, R, Rhat, expfac)
end

Base.size(A::DDAOperator3D) = (3 * A.grid.nvoxels, 3 * A.grid.nvoxels)
Base.eltype(::Type{<:DDAOperator3D}) = ComplexF64
Base.eltype(::DDAOperator3D) = ComplexF64

function Base.getindex(A::DDAOperator3D, row::Int, col::Int)
    1 <= row <= size(A, 1) || throw(BoundsError(A, (row, col)))
    1 <= col <= size(A, 2) || throw(BoundsError(A, (row, col)))
    i = _dda_voxel(row)
    j = _dda_voxel(col)
    a = _dda_component(row)
    b = _dda_component(col)
    if i == j
        return a == b ? 1.0 + 0im : 0.0 + 0im
    end
    iszero(A.alpha[j]) && return 0.0 + 0im
    block = _electric_dipole_alpha_block_3d(
        A.grid.centers[i], A.grid.centers[j], A.k0, A.alpha[j])
    return -block[a, b]
end

@noinline function _dda_operator_field_bigfloat_3d(
        A::DDAOperator3D,
        x::AbstractVector{ComplexF64},
        voxel::Int)
    return setprecision(
            BigFloat, _DDA_ACCUMULATION_FALLBACK_PRECISION) do
        total = SVector{3,Complex{BigFloat}}(ntuple(
            component -> Complex{BigFloat}(
                x[_dda_index(voxel, component)]), 3))
        observation = A.grid.centers[voxel]
        @inbounds for source_voxel in 1:A.grid.nvoxels
            source_voxel == voxel && continue
            alpha = A.alpha[source_voxel]
            iszero(alpha) && continue
            source_field = _read_field_component(x, source_voxel)
            dipole = _alpha_apply_bigfloat_vector_3d(
                alpha, source_field)
            total -= _electric_dipole_value_bigfloat_3d(
                observation,
                A.grid.centers[source_voxel],
                A.k0,
                dipole,
            )
        end
        converted = CVec3(
            ComplexF64(total[1]),
            ComplexF64(total[2]),
            ComplexF64(total[3]),
        )
        all(isfinite, converted) ||
            throw(OverflowError(
                "DDA operator output at voxel $voxel is outside the " *
                "representable ComplexF64 range."))
        return converted
    end
end

@noinline function _dda_adjoint_operator_field_bigfloat_3d(
        A::DDAOperator3D,
        x::AbstractVector{ComplexF64},
        voxel::Int)
    return setprecision(
            BigFloat, _DDA_ACCUMULATION_FALLBACK_PRECISION) do
        total = SVector{3,Complex{BigFloat}}(ntuple(
            component -> Complex{BigFloat}(
                x[_dda_index(voxel, component)]), 3))
        alpha = A.alpha[voxel]
        if !iszero(alpha)
            observation = A.grid.centers[voxel]
            @inbounds for source_voxel in 1:A.grid.nvoxels
                source_voxel == voxel && continue
                source_field = _read_field_component(x, source_voxel)
                conjugated_field = SVector{3,Complex{BigFloat}}(ntuple(
                    component -> conj(Complex{BigFloat}(
                        source_field[component])), 3))
                conjugated_interaction =
                    _electric_dipole_value_bigfloat_3d(
                        observation,
                        A.grid.centers[source_voxel],
                        A.k0,
                        conjugated_field,
                    )
                adjoint_interaction =
                    SVector{3,Complex{BigFloat}}(ntuple(
                        component -> conj(
                            conjugated_interaction[component]), 3))
                total -= _alpha_adjoint_apply_bigfloat_vector_3d(
                    alpha, adjoint_interaction)
            end
        end
        converted = CVec3(
            ComplexF64(total[1]),
            ComplexF64(total[2]),
            ComplexF64(total[3]),
        )
        all(isfinite, converted) ||
            throw(OverflowError(
                "adjoint DDA operator output at voxel $voxel is outside " *
                "the representable ComplexF64 range."))
        return converted
    end
end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::DDAOperator3D,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    length(x) == size(A, 2) || throw(DimensionMismatch("x length must be $(size(A, 2))."))
    length(y) == size(A, 1) || throw(DimensionMismatch("y length must be $(size(A, 1))."))

    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(y, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            y .*= beta_scale
        end
        return y
    end

    xread = Base.mightalias(y, x) ? copy(x) : x
    overwrite = iszero(beta_scale)
    N = A.grid.nvoxels
    # Output voxels are independent, but the matrix-free alloc budget
    # (test/test_mom3d.jl asserts < 1024 bytes) is tighter than the task-spawn
    # overhead of `Threads.@threads` (>= ~1.5 KB even for an empty loop), so
    # threading here would break it. `@inbounds` over the verified-safe indexing
    # (1 <= i,j <= N, components 1..3, xread/y length 3N) keeps the loop at zero
    # allocations while removing per-access bounds checks.
    @inbounds for i in 1:N
        Ei = _read_field_component(xread, i)
        ri = A.grid.centers[i]
        needs_fallback = false
        try
            for j in 1:N
                i == j && continue
                alphaj = A.alpha[j]
                iszero(alphaj) && continue
                next_Ei = Ei - _electric_dipole_alpha_apply_3d(
                    ri, A.grid.centers[j], A.k0, alphaj,
                    _read_field_component(xread, j))
                if !all(isfinite, next_Ei)
                    needs_fallback = true
                    break
                end
                Ei = next_Ei
            end
        catch err
            err isa OverflowError || rethrow()
            needs_fallback = true
        end
        if needs_fallback
            Ei = _dda_operator_field_bigfloat_3d(A, xread, i)
        end

        for a in 1:3
            idx = _dda_index(i, a)
            y[idx] = _dda_scaled_output_3d(
                Ei[a], y[idx], alpha_scale, beta_scale, overwrite, idx)
        end
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::DDAOperator3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

Base.adjoint(A::DDAOperator3D) = DDAAdjointOperator3D(A)
Base.size(A::DDAAdjointOperator3D) = reverse(size(A.parent))
Base.eltype(::Type{<:DDAAdjointOperator3D}) = ComplexF64
Base.eltype(::DDAAdjointOperator3D) = ComplexF64
Base.getindex(A::DDAAdjointOperator3D, row::Int, col::Int) = conj(A.parent[col, row])

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            Aadj::DDAAdjointOperator3D,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    A = Aadj.parent
    length(x) == size(Aadj, 2) || throw(DimensionMismatch("x length must be $(size(Aadj, 2))."))
    length(y) == size(Aadj, 1) || throw(DimensionMismatch("y length must be $(size(Aadj, 1))."))

    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(y, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            y .*= beta_scale
        end
        return y
    end

    xread = Base.mightalias(y, x) ? copy(x) : x
    overwrite = iszero(beta_scale)
    N = A.grid.nvoxels
    # Left serial for the same alloc-budget reason as the forward operator;
    # `@inbounds` is applied over the verified-safe indexing.
    @inbounds for i in 1:N
        Ei = _read_field_component(xread, i)
        alphai = A.alpha[i]
        if !iszero(alphai)
            ri = A.grid.centers[i]
            acc = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
            needs_fallback = false
            try
                for j in 1:N
                    i == j && continue
                    xj = _read_field_component(xread, j)
                    next_acc = acc +
                               _electric_dipole_alpha_adjoint_apply_3d(
                        ri, A.grid.centers[j], A.k0, alphai, xj)
                    if !all(isfinite, next_acc)
                        needs_fallback = true
                        break
                    end
                    acc = next_acc
                end
            catch err
                err isa OverflowError || rethrow()
                needs_fallback = true
            end
            if needs_fallback
                Ei = _dda_adjoint_operator_field_bigfloat_3d(
                    A, xread, i)
            else
                Ei -= acc
            end
        end

        for a in 1:3
            idx = _dda_index(i, a)
            y[idx] = _dda_scaled_output_3d(
                Ei[a], y[idx], alpha_scale, beta_scale, overwrite, idx)
        end
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::DDAAdjointOperator3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

function _validated_dense_dda_system_size(
        grid::VoxelGrid3D,
        max_output_bytes::Integer)
    system_size = try
        Base.Checked.checked_mul(3, grid.nvoxels)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("DDA system dimension overflows Int"))
    end
    matrix_bytes = _checked_array_payload_bytes(
        ComplexF64, system_size, system_size;
        label="dense DDA system matrix")
    _enforce_payload_limit(
        matrix_bytes, max_output_bytes,
        "dense DDA system matrix", "max_output_bytes")
    return system_size
end

"""
    assemble_dda_3d(grid, k0, eps_r;
                    radiative_correction=false,
                    max_output_bytes=2_000_000_000)

Assemble the dense coupled-dipole system

    E_i - sum_{j != i} G_EE(r_i, r_j) alpha_j E_j = E_inc_i

for isotropic complex relative permittivity `eps_r`.

Returns `(A, alpha, epsv)`, where `A` is a `3N x 3N` dense matrix.
"""
function assemble_dda_3d(
        grid::VoxelGrid3D, k0::Real, eps_r;
        radiative_correction::Bool=false,
        max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    k = _finite_positive_k0_3d(k0)
    N = grid.nvoxels
    system_size = _validated_dense_dda_system_size(
        grid, max_output_bytes)
    epsv = _coerce_epsr_material_3d(eps_r, N)
    alpha = _dda_polarizabilities_from_coerced(
        grid, k, epsv, radiative_correction)

    A = Matrix{ComplexF64}(I, system_size, system_size)
    # Each source voxel j writes a disjoint block of columns, so the assembly is
    # threaded over j when worker threads exist.
    if Threads.nthreads() > 1
        Threads.@threads for j in 1:N
            _dda_fill_block_column!(A, grid, alpha, k, j, N)
        end
    else
        for j in 1:N
            _dda_fill_block_column!(A, grid, alpha, k, j, N)
        end
    end

    return A, alpha, epsv
end

# Fill the 3 columns owned by source voxel j. Each j writes disjoint columns,
# so this is safe to run concurrently across j.
@inline function _dda_fill_block_column!(A::Matrix{ComplexF64}, grid::VoxelGrid3D,
                                         alpha, k::Float64, j::Int, N::Int)
    @inbounds begin
        alphaj = alpha[j]
        iszero(alphaj) && return nothing
        rj = grid.centers[j]
        for i in 1:N
            i == j && continue
            block = _electric_dipole_alpha_block_3d(
                grid.centers[i], rj, k, alphaj)
            for c in 1:3
                col = _dda_index(j, c)
                for a in 1:3
                    A[_dda_index(i, a), col] -= block[a, c]
                end
            end
        end
    end
    return nothing
end

"""
    planewave_dda_3d(grid, k_vec, E0, pol)

Evaluate a transverse plane wave at voxel centers:

    E_inc(r) = pol * E0 * exp(-i k_vec dot r)
"""
@noinline function _planewave_phase_bigfloat_dda_3d(
        k_vec::Vec3, center::Vec3, voxel::Int)
    return _source_phase_exact(
        1.0, k_vec, center, -1.0,
        "DDA plane-wave phase at voxel $voxel")
end

function planewave_dda_3d(grid::VoxelGrid3D, k_vec::Vec3, E0, pol)
    khat = _normalized_real_direction_dda_3d(k_vec, "k_vec")
    polv = _as_cvec3(pol, "pol")
    polhat = _normalized_complex_direction_dda_3d(polv, "pol")
    transverse_error = abs(dot(khat, polhat))
    transverse_error <= 1e-10 ||
        error("Plane-wave polarization must be transverse to k_vec; normalized dot=$transverse_error.")

    amp = ComplexF64(E0)
    isfinite(amp) ||
        throw(ArgumentError("E0 must be finite, got $E0."))
    amplitude_requires_exact =
        _source_scaled_vector_requires_fallback(amp, polv)
    field_amplitude = amplitude_requires_exact ? zero(CVec3) : polv * amp
    if !amplitude_requires_exact && !all(isfinite, field_amplitude)
        amplitude_requires_exact = true
    end
    Einc = Vector{CVec3}(undef, grid.nvoxels)
    for j in 1:grid.nvoxels
        center = grid.centers[j]
        needs_exact = amplitude_requires_exact ||
                      _source_phase_requires_fallback(1.0, k_vec, center)
        field = if needs_exact
            _source_scaled_phase_vector_exact(
                amp, polv, 1.0, k_vec, center, -1.0,
                "DDA plane-wave field")
        else
            phase = _source_phase(
                1.0, k_vec, center, -1.0,
                "DDA plane-wave phase")
            ordinary = field_amplitude * phase
            all(isfinite, ordinary) ? ordinary :
                _source_scaled_phase_vector_exact(
                    amp, polv, 1.0, k_vec, center, -1.0,
                    "DDA plane-wave field")
        end
        Einc[j] = field
    end
    return Einc
end

"""
    solve_dda_3d(grid, k0, eps_r, E_inc; radiative_correction=false)

Solve the 3D vector material scattering problem for total electric fields at
voxel centers.
"""
function solve_dda_3d(grid::VoxelGrid3D, k0::Real, eps_r, E_inc::AbstractVector;
                      radiative_correction::Bool=false,
                      solver::Symbol=:direct,
                      max_matrix_bytes::Integer=
                          _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                      tol::Float64=1e-8,
                      maxiter::Int=200,
                      memory::Int=20,
                      verbose::Bool=false,
                      check_gmres_convergence::Bool=true)
    solver == :direct && _validated_dense_dda_system_size(
        grid, max_matrix_bytes)
    rhs = _flatten_fields_3d(E_inc, grid.nvoxels, "E_inc")

    if solver == :direct
        A, alpha, epsv = assemble_dda_3d(
            grid, k0, eps_r;
            radiative_correction=radiative_correction,
            max_output_bytes=max_matrix_bytes,
        )
        fac = _factor_dense_linear_system(
            A, ComplexF64, "direct DDA factorization")
        E_total_flat = _solve_factored_linear_system(
            fac, A, rhs, "direct DDA solution")
        E_total = _unflatten_fields_3d(E_total_flat, grid.nvoxels)
        return DDAResult3D(E_total, _unflatten_fields_3d(rhs, grid.nvoxels),
                           epsv, alpha, A, fac, :direct, nothing,
                           grid, Float64(k0), radiative_correction)
    elseif solver == :gmres
        A = dda_operator_3d(
            grid, k0, eps_r;
            radiative_correction=radiative_correction,
        )
        E_total_flat, stats = solve_gmres(
            A, rhs;
            memory=memory,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
            check_gmres_convergence=check_gmres_convergence,
        )
        E_total = _unflatten_fields_3d(E_total_flat, grid.nvoxels)
        return DDAResult3D(E_total, _unflatten_fields_3d(rhs, grid.nvoxels),
                           A.eps_r, A.alpha, A, nothing, :gmres, stats,
                           grid, Float64(k0), radiative_correction)
    else
        error("Unsupported DDA solver: $solver (expected :direct or :gmres).")
    end
end

"""
    induced_dipoles_dda_3d(result)

Return normalized induced electric dipoles `q_j = p_j / eps0 = alpha_j E_j`.
"""
function induced_dipoles_dda_3d(res::DDAResult3D)
    q = Vector{CVec3}(undef, res.grid.nvoxels)
    for j in 1:res.grid.nvoxels
        q[j] = _scaled_alpha_apply_3d(
            res.alpha[j], res.E_total[j], 1.0,
            "DDA induced dipole", j)
    end
    return q
end

"""
    scattered_field_dda_3d(result, r_obs)

Compute scattered electric field at observation points by summing the radiated
field of all induced dipoles. Observation points must not coincide with voxel
centers.
"""
@noinline function _scattered_field_sum_bigfloat_dda_3d(
        res::DDAResult3D,
        observation::Vec3)
    return setprecision(
            BigFloat, _DDA_ACCUMULATION_FALLBACK_PRECISION) do
        total = zero(SVector{3,Complex{BigFloat}})
        @inbounds for j in 1:res.grid.nvoxels
            iszero(res.alpha[j]) && continue
            dipole = _alpha_apply_bigfloat_vector_3d(
                res.alpha[j], res.E_total[j])
            total += _electric_dipole_value_bigfloat_3d(
                observation, res.grid.centers[j], res.k0, dipole)
        end
        converted = CVec3(
            ComplexF64(total[1]),
            ComplexF64(total[2]),
            ComplexF64(total[3]),
        )
        all(isfinite, converted) ||
            throw(OverflowError(
                "DDA scattered field is outside the representable ComplexF64 range."))
        return converted
    end
end

function _scattered_field_sum_dda_3d(
        res::DDAResult3D,
        observation::Vec3)
    total = zero(CVec3)
    try
        @inbounds for j in 1:res.grid.nvoxels
            iszero(res.alpha[j]) && continue
            contribution = _electric_dipole_alpha_apply_3d(
                observation, res.grid.centers[j], res.k0,
                res.alpha[j], res.E_total[j])
            next_total = total + contribution
            all(isfinite, next_total) ||
                return _scattered_field_sum_bigfloat_dda_3d(
                    res, observation)
            total = next_total
        end
    catch err
        err isa OverflowError || rethrow()
        return _scattered_field_sum_bigfloat_dda_3d(res, observation)
    end
    return total
end

function scattered_field_dda_3d(res::DDAResult3D, r_obs::AbstractVector{Vec3})
    out = Vector{CVec3}(undef, length(r_obs))
    @inbounds for m in eachindex(r_obs)
        out[m] = _scattered_field_sum_dda_3d(res, r_obs[m])
    end
    return out
end

function _farfield_alpha_contribution_bigfloat_dda_3d(
        alpha,
        field::CVec3,
        k::Float64,
        direction::Vec3,
        center::Vec3)
    return setprecision(BigFloat, _POLARIZABILITY_FALLBACK_PRECISION) do
        q = _alpha_apply_bigfloat_vector_3d(alpha, field)
        n = SVector{3,BigFloat}(
            BigFloat(direction[1]),
            BigFloat(direction[2]),
            BigFloat(direction[3]),
        )
        projected = q - n * sum(n[index] * q[index] for index in 1:3)
        phase_argument = BigFloat(k) * sum(
            n[index] * BigFloat(center[index]) for index in 1:3)
        phase = exp(Complex{BigFloat}(0, phase_argument))
        prefactor = BigFloat(k)^2 / (4 * BigFloat(pi))
        value = phase * prefactor * projected
        converted = CVec3(
            ComplexF64(value[1]),
            ComplexF64(value[2]),
            ComplexF64(value[3]),
        )
        all(isfinite, converted) ||
            throw(OverflowError(
                "DDA far-field contribution is outside the representable ComplexF64 range."))
        return converted
    end
end

@inline function _farfield_alpha_contribution_dda_3d(
        alpha,
        field::CVec3,
        k::Float64,
        direction::Vec3,
        center::Vec3,
        projection)
    q = _alpha_apply(alpha, field)
    if all(isfinite, q) &&
       !_alpha_field_product_loses_range_3d(alpha, field)
        phase = _source_phase(
            k, direction, center, 1.0, "DDA far-field phase")
        contribution = phase * (projection * q)
        prefactor, scale_fraction, scale_exponent =
            _farfield_prefactor_dda_3d(k)
        value = iszero(scale_fraction) ?
                prefactor * contribution :
                _scale_farfield_vector_dda_3d(
                    contribution, scale_fraction, scale_exponent)
        all(isfinite, value) && return value
    end
    return _farfield_alpha_contribution_bigfloat_dda_3d(
        alpha, field, k, direction, center)
end

@inline function _dda_farfield_phase_requires_exact(
        res::DDAResult3D,
        direction::Vec3)
    @inbounds for center in res.grid.centers
        _source_phase_requires_fallback(
            res.k0, direction, center) && return true
    end
    return false
end

@noinline function _farfield_sum_bigfloat_dda_3d(
        res::DDAResult3D,
        direction::Vec3)
    return setprecision(
            BigFloat, _DDA_ACCUMULATION_FALLBACK_PRECISION) do
        n = SVector{3,BigFloat}(
            BigFloat(direction[1]),
            BigFloat(direction[2]),
            BigFloat(direction[3]),
        )
        prefactor = BigFloat(res.k0)^2 / (4 * BigFloat(pi))
        total = zero(SVector{3,Complex{BigFloat}})
        @inbounds for j in 1:res.grid.nvoxels
            iszero(res.alpha[j]) && continue
            q = _alpha_apply_bigfloat_vector_3d(
                res.alpha[j], res.E_total[j])
            projected = q - n * sum(
                n[index] * q[index] for index in 1:3)
            phase_argument = BigFloat(res.k0) * sum(
                n[index] * BigFloat(res.grid.centers[j][index])
                for index in 1:3)
            total += exp(Complex{BigFloat}(0, phase_argument)) *
                     prefactor * projected
        end
        converted = CVec3(
            ComplexF64(total[1]),
            ComplexF64(total[2]),
            ComplexF64(total[3]),
        )
        all(isfinite, converted) ||
            throw(OverflowError(
                "DDA far-field amplitude is outside the representable ComplexF64 range."))
        return converted
    end
end

function _farfield_sum_dda_3d(
        res::DDAResult3D,
        direction::Vec3,
        projection)
    _dda_farfield_phase_requires_exact(res, direction) &&
        return _farfield_sum_bigfloat_dda_3d(res, direction)
    total = zero(CVec3)
    try
        @inbounds for j in 1:res.grid.nvoxels
            iszero(res.alpha[j]) && continue
            contribution = _farfield_alpha_contribution_dda_3d(
                res.alpha[j], res.E_total[j], res.k0,
                direction, res.grid.centers[j], projection)
            next_total = total + contribution
            all(isfinite, next_total) ||
                return _farfield_sum_bigfloat_dda_3d(res, direction)
            total = next_total
        end
    catch err
        err isa OverflowError || rethrow()
        return _farfield_sum_bigfloat_dda_3d(res, direction)
    end
    return total
end

"""
    farfield_dda_3d(result, rhat)

Return the far-field amplitude `F(rhat)` such that

    E_scat(r) ~= exp(-i k r) / r * F(rhat)

for unit observation direction `rhat`.
"""
function farfield_dda_3d(res::DDAResult3D, rhat::Vec3)
    n = _normalized_real_direction_dda_3d(rhat, "rhat")
    proj = _I3_DDA - n * transpose(n)
    return _farfield_sum_dda_3d(res, n, proj)
end

function farfield_dda_3d(res::DDAResult3D, rhat::AbstractVector{Vec3})
    return [farfield_dda_3d(res, dir) for dir in rhat]
end
