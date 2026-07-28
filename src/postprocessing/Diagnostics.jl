# Diagnostics.jl — Energy conservation, condition number, and convergence utilities

export radiated_power, projected_power, input_power, energy_ratio, condition_diagnostics
export bistatic_rcs, backscatter_rcs

function _validate_farfield_samples(E_ff::Matrix{<:Number}, grid::SphGrid)
    NΩ = _validate_sph_grid(grid)
    size(E_ff) == (3, NΩ) ||
        throw(DimensionMismatch(
            "E_ff has size $(size(E_ff)), expected (3, $NΩ)"))
    all(isfinite, E_ff) ||
        throw(ArgumentError("E_ff must contain only finite values"))
    return NΩ
end

@inline function _farfield_intensity(E_ff::Matrix{<:Number}, q::Int)
    intensity =
        abs2(E_ff[1, q]) + abs2(E_ff[2, q]) + abs2(E_ff[3, q])
    isfinite(intensity) ||
        throw(OverflowError(
            "far-field intensity overflowed at sample $q"))
    return intensity
end

function _rcs_scale(E0::Real)
    E0_float = try
        Float64(E0)
    catch err
        throw(ArgumentError(
            "E0 must be representable as Float64: $(sprint(showerror, err))"))
    end
    isfinite(E0_float) && !iszero(E0_float) ||
        throw(ArgumentError("E0 must be finite and nonzero, got $E0"))
    root_scale = sqrt(4π) / abs(E0_float)
    (isfinite(root_scale) && root_scale > 0.0) ||
        throw(OverflowError(
            "RCS scale is not representable for E0=$E0"))
    scale = root_scale * root_scale
    (isfinite(scale) && scale > 0.0) ||
        throw(OverflowError(
            "RCS scale is not representable for E0=$E0"))
    return scale
end

"""
    radiated_power(E_ff, grid; eta0=376.730313668)

Compute total radiated power from far-field pattern:
  P_rad = (1/(2η₀)) ∫ |E∞(r̂)|² dΩ ≈ (1/(2η₀)) Σ_q w_q |E∞(r̂_q)|²

The 1/(2η₀) factor converts the far-field electric field intensity to
time-averaged Poynting flux (watts).
"""
function radiated_power(E_ff::Matrix{<:Number}, grid::SphGrid;
                        eta0::Float64=376.730313668)
    NΩ = _validate_farfield_samples(E_ff, grid)
    isfinite(eta0) && eta0 > 0 ||
        throw(ArgumentError("eta0 must be finite and positive, got $eta0"))
    P = 0.0
    @inbounds for q in 1:NΩ
        P += grid.w[q] * _farfield_intensity(E_ff, q)
        isfinite(P) ||
            throw(OverflowError(
                "radiated-power accumulation overflowed at sample $q"))
    end
    power = (P / eta0) / 2
    isfinite(power) ||
        throw(OverflowError("radiated power is non-finite"))
    return power
end

"""
    projected_power(E_ff, grid, pol; mask=nothing)

Compute polarization-projected angular power:
  P = Σ_q w_q |p_q^† E∞(r̂_q)|²

When `mask` is provided, only selected angular samples are included.
This is the discrete quantity represented by `I† Q I` when `Q` is
constructed with the same `pol` and `mask`.
"""
function projected_power(E_ff::Matrix{<:Number}, grid::SphGrid,
                         pol::AbstractMatrix{<:Complex}; mask=nothing)
    NΩ = _validate_farfield_samples(E_ff, grid)
    size(pol) == (3, NΩ) ||
        throw(DimensionMismatch(
            "pol has size $(size(pol)), expected (3, $NΩ)"))
    all(isfinite, pol) ||
        throw(ArgumentError("pol must contain only finite values"))
    if mask !== nothing
        mask isa AbstractVector{Bool} ||
            throw(ArgumentError(
                "mask must be an AbstractVector{Bool}, got $(typeof(mask))"))
        length(mask) == NΩ ||
            throw(DimensionMismatch(
                "mask length $(length(mask)) != $NΩ"))
    end

    P = 0.0
    @inbounds for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        yq = conj(pol[1, q]) * E_ff[1, q] +
             conj(pol[2, q]) * E_ff[2, q] +
             conj(pol[3, q]) * E_ff[3, q]
        isfinite(yq) ||
            throw(OverflowError(
                "polarization projection overflowed at sample $q"))
        P += grid.w[q] * abs2(yq)
        isfinite(P) ||
            throw(OverflowError(
                "projected-power accumulation overflowed at sample $q"))
    end
    return P
end

"""
    input_power(I, v)

Compute the power delivered to the structure:
  P_in = -½ Re(I† v)

For a PEC scatterer with Z I = v, this is the power extracted from the
incident field by the induced currents.
"""
function input_power(I::Vector{<:Number}, v::Vector{<:Number})
    length(I) == length(v) ||
        throw(DimensionMismatch(
            "I length $(length(I)) != v length $(length(v))"))
    !isempty(I) ||
        throw(ArgumentError("I and v must contain at least one entry"))
    all(isfinite, I) ||
        throw(ArgumentError("I must contain only finite values"))
    all(isfinite, v) ||
        throw(ArgumentError("v must contain only finite values"))
    power = -0.5 * real(dot(I, v))
    isfinite(power) ||
        throw(OverflowError("input-power evaluation overflowed"))
    return power
end

"""
    energy_ratio(I, v, E_ff, grid; eta0=376.730313668)

Compute the ratio P_rad / P_in as an energy conservation diagnostic.
For a lossless PEC structure, this should be ≈ 1.
For an impedance sheet with Re(Z_s) > 0, P_rad/P_in < 1 (absorbed power).
"""
function energy_ratio(I::Vector{<:Number}, v::Vector{<:Number},
                      E_ff::Matrix{<:Number}, grid::SphGrid;
                      eta0::Float64=376.730313668)
    P_in  = input_power(I, v)
    P_rad = radiated_power(E_ff, grid; eta0=eta0)
    !iszero(P_in) ||
        throw(DomainError(
            P_in, "energy ratio is undefined for zero input power"))
    ratio = P_rad / P_in
    isfinite(ratio) ||
        throw(OverflowError("energy ratio is non-finite"))
    return ratio
end

"""
    condition_diagnostics(Z)

Return condition number and singular value extremes of the MoM matrix.
"""
function condition_diagnostics(Z::Matrix{<:Number})
    (size(Z, 1) > 0 && size(Z, 2) > 0) ||
        throw(ArgumentError(
            "Z must have at least one row and one column, got size $(size(Z))"))
    all(isfinite, Z) ||
        throw(ArgumentError("Z must contain only finite values"))
    svs = svdvals(Z)
    all(isfinite, svs) ||
        throw(OverflowError(
            "singular-value computation produced non-finite values"))
    sv_max = first(svs)
    sv_min = last(svs)
    κ = iszero(sv_min) ? Inf : sv_max / sv_min
    return (cond=κ, sv_max=sv_max, sv_min=sv_min)
end

"""
    bistatic_rcs(E_ff; E0=1.0)

Compute bistatic radar cross section samples from far-field amplitudes:
  σ(r̂_q) = 4π |E∞(r̂_q)|² / |E0|²

Returns a real vector of length `NΩ` in linear units (m²).
"""
function bistatic_rcs(E_ff::Matrix{<:Number}; E0::Real=1.0)
    size(E_ff, 1) == 3 ||
        throw(DimensionMismatch(
            "E_ff has $(size(E_ff, 1)) rows, expected 3"))
    all(isfinite, E_ff) ||
        throw(ArgumentError("E_ff must contain only finite values"))
    NΩ = size(E_ff, 2)
    scale = _rcs_scale(E0)
    σ = zeros(Float64, NΩ)
    @inbounds for q in 1:NΩ
        σ[q] = scale * _farfield_intensity(E_ff, q)
        isfinite(σ[q]) ||
            throw(OverflowError(
                "bistatic RCS overflowed at sample $q"))
    end
    return σ
end

"""
    backscatter_rcs(E_ff, grid, k_inc_hat; E0=1.0)

Return monostatic/backscatter RCS for a plane-wave incidence direction
`k_inc_hat` (unit propagation direction). The backscatter direction is
`-k_inc_hat`, mapped to the nearest sample on `grid`.

Returns a named tuple:
`(sigma, index, theta, phi, angular_error_deg)`.
"""
function backscatter_rcs(E_ff::Matrix{<:Number}, grid::SphGrid,
                         k_inc_hat::Vec3; E0::Real=1.0)
    NΩ = _validate_farfield_samples(E_ff, grid)
    all(isfinite, k_inc_hat) ||
        throw(ArgumentError("k_inc_hat components must be finite"))
    scale = _rcs_scale(E0)

    khat = _validated_farfield_direction(k_inc_hat)
    r_back = -khat

    best_idx = 1
    best_dot = -Inf
    @inbounds for q in 1:NΩ
        d = r_back[1] * grid.rhat[1, q] +
            r_back[2] * grid.rhat[2, q] +
            r_back[3] * grid.rhat[3, q]
        if d > best_dot
            best_dot = d
            best_idx = q
        end
    end

    ang_err = acos(clamp(best_dot, -1.0, 1.0)) * 180 / π
    sigma = scale * _farfield_intensity(E_ff, best_idx)
    isfinite(sigma) ||
        throw(OverflowError("backscatter RCS overflowed"))

    return (
        sigma = sigma,
        index = best_idx,
        theta = grid.theta[best_idx],
        phi = grid.phi[best_idx],
        angular_error_deg = ang_err,
    )
end
