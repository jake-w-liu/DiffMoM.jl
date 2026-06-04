# Diagnostics.jl — Energy conservation, condition number, and convergence utilities

export radiated_power, projected_power, input_power, energy_ratio, condition_diagnostics
export bistatic_rcs, backscatter_rcs

"""
    radiated_power(E_ff, grid; eta0=376.730313668)

Compute total radiated power from far-field pattern:
  P_rad = (1/(2η₀)) ∫ |E∞(r̂)|² dΩ ≈ (1/(2η₀)) Σ_q w_q |E∞(r̂_q)|²

The 1/(2η₀) factor converts the far-field electric field intensity to
time-averaged Poynting flux (watts).
"""
function radiated_power(E_ff::Matrix{<:Number}, grid::SphGrid;
                        eta0::Float64=376.730313668)
    NΩ = length(grid.w)
    P = 0.0
    for q in 1:NΩ
        Eq = E_ff[:, q]
        P += grid.w[q] * real(dot(Eq, Eq))
    end
    return P / (2 * eta0)
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
    NΩ = length(grid.w)
    P = 0.0
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        yq = dot(pol[:, q], E_ff[:, q])
        P += grid.w[q] * real(conj(yq) * yq)
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
    return -0.5 * real(dot(I, v))
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
    return P_rad / P_in
end

"""
    condition_diagnostics(Z)

Return condition number and singular value extremes of the MoM matrix.
"""
function condition_diagnostics(Z::Matrix{<:Number})
    svs = svdvals(Z)
    return (cond = svs[1] / svs[end],
            sv_max = svs[1],
            sv_min = svs[end])
end

"""
    bistatic_rcs(E_ff; E0=1.0)

Compute bistatic radar cross section samples from far-field amplitudes:
  σ(r̂_q) = 4π |E∞(r̂_q)|² / |E0|²

Returns a real vector of length `NΩ` in linear units (m²).
"""
function bistatic_rcs(E_ff::Matrix{<:Number}; E0::Real=1.0)
    abs2(E0) > 0 || error("E0 must be nonzero for RCS normalization")
    NΩ = size(E_ff, 2)
    σ = zeros(Float64, NΩ)
    scale = 4π / abs2(E0)
    for q in 1:NΩ
        Eq = E_ff[:, q]
        σ[q] = scale * real(dot(Eq, Eq))
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
    khat = k_inc_hat / norm(k_inc_hat)
    r_back = -khat

    best_idx = 1
    best_dot = -Inf
    NΩ = length(grid.w)
    for q in 1:NΩ
        rq = Vec3(grid.rhat[:, q])
        d = dot(r_back, rq)
        if d > best_dot
            best_dot = d
            best_idx = q
        end
    end

    σ = bistatic_rcs(E_ff; E0=E0)
    ang_err = acos(clamp(best_dot, -1.0, 1.0)) * 180 / π

    return (
        sigma = σ[best_idx],
        index = best_idx,
        theta = grid.theta[best_idx],
        phi = grid.phi[best_idx],
        angular_error_deg = ang_err,
    )
end
